'''
Stable version for the rule learning model on training and testing 
'''
import os
import pickle
import sys
sys.path.append(os.getcwd())
import sys
import numpy as np
import pandas as pd
import matplotlib  
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from datetime import datetime
from neuralBack.dforl_torch import *
from rule_learning_original.code.dkm_pytorch_stable.compgraph import DkmCompGraph
from first_order_ts import *
import rule_learning_original.code.spec.specific_end_v2 as specs
from termcolor import colored

class EndToEndTimeSeries(TimeSeries):
    def __init__(self, rule_learning_method='ripper', train_data_path='', test_data_path='', result_name='', task_name='', father_path='', unit=1, low=4, high=11, max_plot = 50, min_precision = 0.8, min_recall = 0.7, max_inter = 10,multi_class_label=None, n_clustering = 5,model_type = 'time_period', device='cuda:1', test_flag = True,minimum_precision=0.3,data_type = None) -> None:
        super().__init__(rule_learning_method, train_data_path, test_data_path, result_name, task_name, father_path, unit, low, high, max_plot)
        self.n_clustering = n_clustering
        self.max_inter = max_inter
        self.minimum_precision = min_precision
        self.min_recall = min_recall
        self.subsequence_length = specs.subsequence_length # minimum sequence length for clustering method 
        self.stride = specs.stride # for clustering 
        self.data_type = data_type
        self.image_row = specs.image_row
        self.image_column = specs.image_column
        
        # read data and obtain the length 
        # self.train_path = train_data_path
        # data = pd.read_csv(self.train_path)
        # length = data.shape[1]-1
        # if length > 100:
        #     number_period = 2 * specs.number_period
        # else:
        #     number_period = specs.number_period
        # self.period_length = int(length/number_period)
        
        
        self.period_length = specs.period_length # each head subsequence position indicates the time period of the whole subsequence 
        self.data_process()
        self.model_type= model_type
        self.device = device
        self.threshold = 0.5
        self.time_stamp = datetime.now()
        self.models_path = f'{father_path}/model/best_logic_model_{self.time_stamp}.pt'
        self.cg_model = f'{father_path}/model/best_cg_model_{self.time_stamp}.pt'
        # output from clustering model, used to interpret the rules 
        self.normalized_weighted_dist = None
        color_basic = ['-r','-g','-b','-y','-c','-m'] * 3
        color_re_tiem = 100
        self.color = self.prepare_color(color_basic, color_re_tiem)
        self.lambda_ = 1
        self.test_flag = test_flag

    
    def prepare_color(self, color_basic, color_re_tiem):
        color = []
        for i in range(color_re_tiem):
            color.extend(color_basic)
        return color
        
    
    def logic(self, clustering_possibility, number_batch_sample, y, logic_model):
        if self.model_type == 'no_time':
            return self.run_logic(clustering_possibility, number_batch_sample, y, logic_model)
        elif self.model_type == 'time_period':
            return self.run_logic_with_time(clustering_possibility, number_batch_sample, y, logic_model)
        else:
            raise('model type not supported')
    
    def move_label_to_first_row_and_normalization(self, data_path=None):
        '''
        Move the label to the first row
        '''
        data = pd.read_csv(data_path)
        if data[self.label_name].unique().shape[0] > 2:
            raise('The label number is not binary')
        if data[self.label_name].max() > 1:
            data[self.label_name] = data[self.label_name].apply(lambda x: 0 if x > 1 else 1)
        # drop nan rows
        data = data.dropna(axis=0).reset_index(drop=True)
        label = data[self.label_name]
        data = data.drop(columns=[self.label_name])
        
        data_array = np.array(data)
        # normalization the data
        nor_data = (data_array - data_array.min())/(data_array.max() - data_array.min())
        # z-normalization data above each row 
        # nor_data = (data_array - data_array.mean(axis=1, keepdims=True)) / data_array.std(axis=1, keepdims=True)
        
        
        nor_data = pd.DataFrame(nor_data, columns=data.columns)
        nor_data = pd.concat([label, nor_data], axis=1)
        # data.to_csv(data_path, index=False)
        return nor_data

        
    def data_process(self):
        self.row_train = self.move_label_to_first_row_and_normalization(self.train_path)
        self.row_test = self.move_label_to_first_row_and_normalization(self.test_path)
        self.data_map_train = self.split_subsequence(self.row_train, length=self.subsequence_length, stride=self.stride)
        self.data_map_test = self.split_subsequence(self.row_test, length=self.subsequence_length, stride=self.stride)
        all_sequence_data_train  = []
        all_sequence_data_test = []
        y_train = []
        y_test = []
        for i in range(len(self.data_map_train)):
            all_sequence_data_train.extend(self.data_map_train[i][1:])
            y_train.append(self.data_map_train[i][0])
        self.all_sequence_data_train = np.array(all_sequence_data_train)
        self.y_train = np.array(y_train, dtype=np.float32)

        for i in range(len(self.data_map_test)):
            all_sequence_data_test.extend(self.data_map_test[i][1:])
            y_test.append(self.data_map_test[i][0])
        self.all_sequence_data_test = np.array(all_sequence_data_test)
        self.y_test = np.array(y_test, dtype=np.float32)
        
        # number_time_points = self.all_sequence_data.shape[1]
        # self.number_to_filled = self.minimumSequenceLength - (number_time_points % self.minimumSequenceLength)
        # number_time_points = number_time_points + self.number_to_filled
        
        # calculate the number of periods for consider the time period in data 
        if self.all_sequence_data_train.shape[1] % self.period_length != 0:
            self.number_intervals = int(self.all_sequence_data_train.shape[1]/self.period_length) + 1
        else:
            self.number_intervals = int(self.all_sequence_data_train.shape[1]/self.period_length)
        

    def run_loop(self):
        x_train = np.array(self.all_sequence_data_train, dtype=np.float32)
        x_train = torch.tensor(x_train)
        y_train = torch.tensor(self.y_train)
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        
        x_test = torch.tensor(np.array(self.all_sequence_data_test, dtype=np.float32))
        y_test = torch.tensor(self.y_test)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        
        # define data for training
        supervised_data_train = torch.utils.data.DataLoader(train_dataset, batch_size=specs.batch_size, shuffle=False, num_workers=0)
        supervised_data_test = torch.utils.data.DataLoader(test_dataset, batch_size=specs.batch_size, shuffle=False, num_workers=0)

        
        # only x data 
        all_sequence_train = np.array(self.all_sequence_data_train, dtype=np.float32).reshape(-1, self.subsequence_length)
        all_sequence_train = torch.tensor(all_sequence_train)
        all_sequences_loader_train = torch.utils.data.DataLoader(all_sequence_train, batch_size=specs.ae_batch_size, shuffle=False, num_workers=0)
        
        all_sequence_test = np.array(self.all_sequence_data_test, dtype=np.float32).reshape(-1, self.subsequence_length)
        all_sequence_test = torch.tensor(all_sequence_test)
        all_sequences_loader_test = torch.utils.data.DataLoader(all_sequence_test, batch_size=specs.batch_size, shuffle=False, num_workers=0)
        
        writer = SummaryWriter(f'demo/{self.time_stamp}/')
        
        
        # define autoencoder model
        hidden_1_size = 500
        hidden_2_size = 500
        hidden_3_size = 2000
        embedding_size = self.n_clustering
        assert all_sequence_train.shape[1] == all_sequence_test.shape[1]
        input_size = all_sequence_train.shape[1]
        dimensions = [hidden_1_size, hidden_2_size, hidden_3_size, embedding_size, # Encoder layer dimensio
              hidden_3_size, hidden_2_size, hidden_1_size, input_size] # Decoder layer dimensions
        activations = [torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU(), None, # Encoder layer activations
                torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU(), None] # Decoder layer activations
        names = ['enc_hidden_1', 'enc_hidden_2', 'enc_hidden_3', 'embedding', # Encoder layer names
            'dec_hidden_1', 'dec_hidden_2', 'dec_hidden_3', 'output'] # Decoder layer names
        
        # define dkm and autoencoder model 
        cg = DkmCompGraph([dimensions, activations, names], self.n_clustering, self.lambda_, input_size, device=self.device)
        n_samples_train = all_sequence_train.shape[0]
        n_samples_test = all_sequence_test.shape[0]
        
        # doing pretraining 
        if specs.pre_training:
            pre_para = list(cg.encoder.parameters()) + list(cg.decoder.parameters())
            optimizer = torch.optim.Adam(pre_para , lr=specs.learning_rate_ae)
            print('Starting autoencoder pretraining')
            
            embeddings_train = np.zeros((n_samples_train, embedding_size), dtype=float)
            embeddings_test = np.zeros((n_samples_train, embedding_size), dtype=float)
            for epoch in range(specs.pretrain_epoch):
                print("Pretraining epoch: ", epoch)
                for i, batch in enumerate(all_sequences_loader_train):
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    ae_loss, embedding_, output = cg.get_reconstruction_loss(batch)
                    ae_loss.backward()
                    optimizer.step()
                    
                    for j in range(0, len(batch)):
                        embeddings_train[j+i,:] = embedding_[j,:].detach().cpu().numpy()
                    print('ae_loss', ae_loss.item())   
        
            
        print('[pretraining done]')
        kmeans_model = KMeans(n_clusters=self.n_clustering, init="k-means++").fit(embeddings_train)
        print('[kmeans done]')
        print('K-Means prediction for embeddings:')
        print(kmeans_model.predict(embeddings_train))
        cg.cluster_rep = torch.nn.Parameter(torch.tensor(kmeans_model.cluster_centers_, dtype=torch.float32, device=self.device), requires_grad=True)
        
        
        # # doing finetuning
        distances = np.zeros((self.n_clustering, n_samples_train), dtype=float)
        optimizer = torch.optim.Adam(cg.parameters(), lr = specs.learning_rate_dkm)
                
        for epoch_index in range(specs.finetuing_epoch):
            running_loss = 0
            for i, x in enumerate(all_sequences_loader_train):
                x = x.to(self.device)
                optimizer.zero_grad()
                loss, stack_dist, ae_loss, kmean_loss, weighted_dist = cg(x, alpha=specs.alphas)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            for j in range(len(x)):
                distances[:, i+j] = stack_dist.detach().cpu().numpy()[:, j]
            last_loss = running_loss/len(all_sequences_loader_train)
            print(f'Epoch {epoch_index} loss {last_loss}')
            writer.add_scalar('Loss/Total', last_loss, epoch_index)
        print('[finetune embedding done]')
            
        
        if self.model_type == 'no_time':
            logic_model = DeepRuleLayer_v2(in_size=self.n_clustering, rule_number=2, output_rule_file=self.result_name+'/res.md', t_relation='', task='',t_arity='', device=self.device,time_stamp='').to(self.device)
        elif self.model_type == 'time_period':
            logic_model = DeepRuleLayer_v2(in_size=self.number_intervals * self.n_clustering, rule_number=2, output_rule_file=self.result_name+'/res.md', t_relation='', task='',t_arity='', device=self.device,time_stamp='').to(self.device)
        
        
        optimizer = torch.optim.Adam(cg.parameters(),specs.learning_rate_dkm)
        learning_step = 0
        for epoch_index in range(specs.finetuing_epoch_outter):
            running_loss = 0
            running_ae_loss = 0
            runn_dkm_loss = 0
            
            for i, (x,y) in enumerate(supervised_data_train):
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                number_batch_sample = x.shape[0]
                sequence_data = x.reshape(-1, self.subsequence_length)
                dkm_loss, stack_dist, ae_loss, kmean_loss, cluster_possibility = cg(sequence_data, specs.alphas)
                dkm_loss.backward()
                optimizer.step()
                
                running_loss += dkm_loss.item()
                running_ae_loss += ae_loss.item()
                runn_dkm_loss += kmean_loss.item()
                
            last_loss = running_loss/len(supervised_data_train)
            running_ae_loss = running_ae_loss/len(supervised_data_train)
            runn_dkm_loss = runn_dkm_loss/len(supervised_data_train)
            
            print(f'Epoch {epoch_index} loss {round(last_loss,2)}, ae loss {round(running_ae_loss,2)}, center loss {round(runn_dkm_loss,2)}')
            
        #! from AD -> (DKM) -> Logic version: First train DKM, then train Logic, and train DKM, and train logic 
        logic_optimizer = torch.optim.Adam([
            {'params':logic_model.parameters(), 'lr': specs.learning_rate_logic},
            {'params':cg.parameters(), 'lr': specs.learning_rate_dkm}
            ])
        
        best_acc_train = 0
        self.best_acc_test = 0
        best_acc_plot_train = 0
        best_acc_plot_test = 0
        
        for logic_epoch in range(specs.finetuing_logic_epoch):
            runn_logic_loss = 0
            running_correct = 0
            running_ae_loss = 0 
            running_loss = 0
            runn_dkm_loss = 0
            runnacc = 0
            
            runn_logic_loss_test = 0
            running_correct_test = 0
            running_ae_loss_test = 0 
            running_loss_test = 0
            runn_dkm_loss_test = 0
            runnacc_test = 0
            
            for i, (x,y) in enumerate(supervised_data_train):
                x = x.to(self.device)
                y = y.to(self.device)
                number_batch_sample = x.shape[0]
                logic_optimizer.zero_grad()
                
                # update the clustering model
                sequence_data = x.reshape(-1, self.subsequence_length)
                dkm_loss, stack_dist, ae_loss, kmean_loss, cluster_possibility = cg(sequence_data, specs.alphas)
                # update the logic model
                logic_loss, inferred_y,_,_ = self.logic(cluster_possibility, number_batch_sample, y, logic_model)
                all_loss = dkm_loss + logic_loss
                all_loss.backward()
                logic_optimizer.step()
                runn_logic_loss += logic_loss.item()
                running_loss += all_loss.item()
                running_ae_loss += ae_loss.item()
                runn_dkm_loss += kmean_loss.item()
                # compute acc 
                threshold_inferred_y = torch.where(inferred_y>self.threshold, torch.tensor(1.0).to(self.device), torch.tensor(0.0).to(self.device))
                correct = torch.sum(torch.eq(y, threshold_inferred_y)).item()
                running_correct += correct
            running_loss = running_loss/len(supervised_data_train)
            runn_logic_loss = runn_logic_loss/len(supervised_data_train)
            running_ae_loss = running_ae_loss/len(supervised_data_train)
            runn_dkm_loss = runn_dkm_loss/len(supervised_data_train)
            runnacc = running_correct/(self.y_train.shape[0])
            
            if self.test_flag == True:
                # test the data 
                for i, (x,y) in enumerate(supervised_data_test):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    number_batch_sample = x.shape[0]
                    # update the clustering model
                    sequence_data = x.reshape(-1, self.subsequence_length)
                    dkm_loss, stack_dist, ae_loss, kmean_loss, cluster_possibility = cg(sequence_data, specs.alphas)
                    # update the logic model
                    logic_loss, inferred_y,_,_ = self.logic(cluster_possibility, number_batch_sample, y, logic_model)
                    all_loss = dkm_loss + logic_loss
                    runn_logic_loss += logic_loss.item()
                    running_loss += all_loss.item()
                    running_ae_loss += ae_loss.item()
                    runn_dkm_loss += kmean_loss.item()
                    # compute acc 
                    threshold_inferred_y = torch.where(inferred_y>self.threshold, torch.tensor(1.0).to(self.device), torch.tensor(0.0).to(self.device))
                    correct = torch.sum(torch.eq(y, threshold_inferred_y)).item()
                    running_correct_test += correct
                
                running_loss_test = running_loss/len(supervised_data_test)
                runn_logic_loss_test = runn_logic_loss/len(supervised_data_test)
                running_ae_loss_test = running_ae_loss/len(supervised_data_test)
                runn_dkm_loss_test = runn_dkm_loss/len(supervised_data_test)
                runnacc_test = running_correct_test/(self.y_test.shape[0])
            
            if self.test_flag == False:
                if runnacc > best_acc_train:
                    best_acc_train = runnacc
                    torch.save(logic_model.state_dict(), self.models_path)
                    torch.save(cg.state_dict(), self.cg_model)
            else:
                if runnacc > best_acc_train:
                    best_acc_train = runnacc
                if runnacc_test > self.best_acc_test:
                    self.best_acc_test = runnacc_test
                    # for param in logic_model.parameters():
                    #     print(param)
                    # for param in cg.parameters():
                    #     print(param)
                    torch.save(logic_model.state_dict(), self.models_path)
                    torch.save(cg.state_dict(), self.cg_model)
                
            print(f'Epoch Logic {logic_epoch} loss {round(running_loss,2)}, ae loss {round(running_ae_loss,2)}, center loss {round(runn_dkm_loss,2)}, logic {round(runn_logic_loss,2)}, acc {round(runnacc,2)}, best acc {round(best_acc_train,2)}, test loss {round(running_loss_test,2)}, test ae loss {round(running_ae_loss_test,2)}, test center loss {round(runn_dkm_loss_test,2)}, test logic {round(runn_logic_loss_test,2)}, test acc {round(runnacc_test,2)}, best test acc {round(self.best_acc_test,2)}.')  
            
            # if the acc on the test data is 1, then stop training
            if self.best_acc_test == 1 and self.test_flag == True:
                break
            if best_acc_train == 1 and self.test_flag == False:
                break
            
            writer.add_scalar('Train/Loss/Total', running_loss, learning_step)
            writer.add_scalar('Train/Loss/AE', running_ae_loss, learning_step+specs.pretrain_epoch)
            writer.add_scalar('Train/Loss/Center', runn_dkm_loss, learning_step)
            writer.add_scalar('Train/Loss/Logic', runn_logic_loss, learning_step)
            writer.add_scalar('Train/Best_ACC', best_acc_train, learning_step)
            
            writer.add_scalar('Test/Loss/Total', running_loss_test, learning_step)
            writer.add_scalar('Test/Loss/AE', running_ae_loss_test, learning_step+specs.pretrain_epoch)
            writer.add_scalar('Test/Loss/Center', runn_dkm_loss_test, learning_step)
            writer.add_scalar('Test/Loss/Logic', runn_logic_loss_test, learning_step)
            writer.add_scalar('Test/Best_ACC', self.best_acc_test, learning_step)
            
            learning_step += 1
        # begin to interpret the rules 
        # self.test_rule()
        if self.best_acc_test > best_acc_plot_test and self.test_flag == True:
            print('ploting becuase acc is improved')
            self.interpret_rules(logic_model,cg)
            best_acc_plot_test = self.best_acc_test
            
        if best_acc_train > best_acc_plot_train and self.test_flag == False:
            print('ploting becuase acc is improved')
            self.interpret_rules(logic_model,cg)
            best_acc_plot_train = best_acc_train
            
        if best_acc_plot_test == 1 and self.test_flag == True:
            print('best acc is 1')

        if best_acc_plot_train == 1 and self.test_flag == False:
            print('best acc is 1')
        
        print('FineTunning logic model done')
        
        # # record test acc by neural networks 
        # with open(f'{self.result_name}/best_acc_test.txt', 'w') as f:
        #     f.write(self.time_stamp.strftime("%Y-%m-%d %H:%M:%S") + '\n')
        #     f.write('acc neural test',str(self.best_acc_test))
        #     f.close()
        
        return self.best_acc_test
        
        
    def run_ae(self, dataload, optimizer, ae_model, mse, writer):
        for epoch_index in range(specs.pretrain_epoch):
            running_loss = 0
            for x in dataload:
                x = x.to(self.device)
                optimizer.zero_grad()
                embedding, output = ae_model(x)
                loss = mse(output, x)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            last_loss = running_loss/len(dataload)
            print('AE loss',last_loss)
            writer.add_scalar('Loss/AE', last_loss, epoch_index)
        return 0 
        
    
    def run_dkm(self):
        return 0
    
    def run_logic(self, weighted_dist, number_batch_sample, y, logic_model):
        #compute logic loss 
        weighted_dist_logic = torch.reshape(weighted_dist, (number_batch_sample, -1, self.n_clustering))
        summed_weighted_dist = torch.sum(weighted_dist_logic, dim=1)
        normalized_weighted_dist = summed_weighted_dist/(torch.max(summed_weighted_dist, dim=1)[0].reshape(-1,1)+0.01)
        
        inferred_y = logic_model(normalized_weighted_dist).reshape(-1)
        # TODO First try MSE here
        logic_loss = torch.nn.functional.mse_loss(inferred_y, y)
        return logic_loss, inferred_y
    
    def run_logic_with_time(self, clustering_possibility, number_batch_sample, y, logic_model):
        clustering_possibility_t = torch.transpose(clustering_possibility, 0, 1)
        weighted_dist_logic = torch.reshape(clustering_possibility_t, (number_batch_sample, -1, self.n_clustering)) # the first is the number of sample, the second is the number of all sequence, and the third number is the possible of each clustering
        
        if weighted_dist_logic.shape[1] % self.period_length != 0:
            number_to_filed = self.period_length - (weighted_dist_logic.shape[1] % self.period_length)
        else:
            number_to_filed = 0

        # if the number of intervals is not enough, we need to fill the matrix with zeros
        weighted_dist_logic = torch.cat([weighted_dist_logic, torch.zeros(number_batch_sample, number_to_filed ,self.n_clustering).to(self.device)], dim=1)
        self.weighted_dist_logic = torch.reshape(weighted_dist_logic, (number_batch_sample, -1, self.period_length, self.n_clustering)) # the first dimension is the number of sample, the second is the number of the periods, the third is the sequence number in each period, the last is the clustering number in each sequences
        
        # TODO this relazation is important to write in the paper
        summed_weighted_dist = torch.sum(self.weighted_dist_logic, dim=2)
        # include the clustering information in each interval based all contributed sequences
        
        # get normazlied weighted dist
        sum_in_second_dim = torch.sum(summed_weighted_dist, dim=2, keepdim=True)
        normalized_weighted_dist = summed_weighted_dist/(sum_in_second_dim)

        # softmax_probability_for_period = torch.nn.functional.softmax(summed_weighted_dist, dim=2)
        # the matrix would be interval#1cluster#1, interval#1cluster#2, interval#2cluster#1, interval#2cluster#2..., interval#ncluster#1, interval#ncluster#2  
        self.normalized_weighted_dist = torch.reshape(normalized_weighted_dist, (number_batch_sample, -1))
        
        
        # summed_weighted_dist = torch.sum(weighted_dist_logic, dim=1)
        # normalized_weighted_dist = summed_weighted_dist/(torch.max(summed_weighted_dist, dim=1)[0].reshape(-1,1)+0.01)
        inferred_y = logic_model(self.normalized_weighted_dist).reshape(-1)
        # TODO First try MSE here
        logic_loss = torch.nn.functional.mse_loss(inferred_y, y)
        return logic_loss, inferred_y, self.weighted_dist_logic, self.normalized_weighted_dist
    
    def test_rule(self):
        with open(f'{self.result_name}/mex_obj_2024-07-29 15:42:10.734293.pkl', 'rb') as f:
            rule = pickle.load(f)
            f.close()
        self.plot(rule)
        
    def interpret_rules(self, logic_model:DeepRuleLayer_v2,cg:DkmCompGraph):
        '''
        in the test data: test_y = self.y_test
        in the train data: test_y = self.y_train
        '''
        # load best model 
        logic_model.load_state_dict(torch.load(self.models_path))
        # for param in logic_model.parameters():
        #     print(param)
        
        cg.load_state_dict(torch.load(self.cg_model))
        # for param in cg.parameters():
        #     print(param)
        
        # plot the metrics and rules after each clustering method
        all_columns = []
        for i in range(self.number_intervals):
            for j in range(self.n_clustering):
                all_columns.append(f'interval_{i}_cluster_{j}')
        all_columns.append('label')
        # ! obtain the data clustering index, during the interpretable process, the self.normalized_weighted_dist is generated from test datasets
        
        # step 1 make a inference based all test data 
        test_x = torch.tensor(self.all_sequence_data_test, dtype=torch.float32).to(self.device)
        test_y=torch.tensor(self.y_test,dtype=torch.float32).to(self.device)
        test_data = torch.utils.data.TensorDataset(test_x, test_y)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)
        
        # initial all normalized weighted dist
        normalized_weighted_dist_all_test = []
        weighted_dist_logic_all_test = []
        for i, (x,y) in enumerate(test_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            number_batch_sample = x.shape[0]
            sequence_data = x.reshape(-1, self.subsequence_length)
            dkm_loss, stack_dist, ae_loss, kmean_loss, cluster_possibility = cg(sequence_data, specs.alphas)
            logic_loss, inferred_y,weighted_dist_logic, normalized_weighted_dist  = self.logic(cluster_possibility, number_batch_sample, y, logic_model)
            list_normal = normalized_weighted_dist.detach().cpu().numpy().tolist()
            list_weight = weighted_dist_logic.detach().cpu().numpy().tolist()
            normalized_weighted_dist_all_test.extend(list_normal)
            weighted_dist_logic_all_test.extend(list_weight)
            print('interpret testing',i)
            
            
        
        # number_batch_sample = test_x.shape[0]
        # sequence_data = test_x.reshape(-1, self.subsequence_length)
        # dkm_loss, stack_dist, ae_loss, kmean_loss, cluster_possibility = cg(sequence_data, specs.alphas)
        # logic_loss, inferred_y,weighted_dist_logic, normalized_weighted_dist  = self.logic(cluster_possibility, number_batch_sample, test_y, logic_model)
        
        
        # ! prepare for the interpretable tensors 
        normalized_weighted_dist = np.array(normalized_weighted_dist_all_test)
        weighted_dist_logic = np.array(weighted_dist_logic_all_test)
        # interpret the trained clustering mdoels
        # normalized_weighted_dist = self.normalized_weighted_dist.detach().cpu().numpy()
        # maximum is 1, other is 0 
        # todo opinion 1, choose the maximum clutering as 1, others as 0 [We choose this]
        normalized_weighted_dist_to_cluster = normalized_weighted_dist.reshape(normalized_weighted_dist.shape[0], -1, self.n_clustering)
        normalized_weighted_dist_threshold = np.where(normalized_weighted_dist_to_cluster == normalized_weighted_dist_to_cluster.max(axis=2)[:,:,None], 1, 0)
        threshold_normalized_weighted_dist = normalized_weighted_dist_threshold.reshape(normalized_weighted_dist.shape[0], -1)
        # todo opinion 2, choose the clustering with larger than 0.5 as 1, others as 0
        # threshold_normalized_weighted_dist = np.where(normalized_weighted_dist>self.threshold, 1, 0)
        test_y = test_y.detach().cpu().numpy()
        threshold_normalized_weighted_dist = np.concatenate((threshold_normalized_weighted_dist, test_y.reshape(-1,1)), axis=1)
        test_data = pd.DataFrame(threshold_normalized_weighted_dist, columns=all_columns)
        
        # interpret rules based on the test data
        rule_set = logic_model.interpret(test_data, precision=1000)
        mex_obj = logic_model.check_metric(test_data, rule_set, self.minimum_precision, print_result= True, update_parameters=False, single_check=True, append_info=f'acc_neural: {self.best_acc_test}')
        # obtain the rules 
        if len(mex_obj['rule_set']) == 0:
            return None
        else:
            with open(f'{self.result_name}/mex_obj_{self.time_stamp}.pkl', 'wb') as f:
                pickle.dump(mex_obj, f)
                f.close()
            if self.test_flag == False:
                self.plot(mex_obj, self_row_data=self.row_train, all_sequence_data=self.all_sequence_data_train, normalized_weighted_dist=normalized_weighted_dist)
            else:
                self.plot(mex_obj, self_row_data=self.row_test, all_sequence_data=self.all_sequence_data_test, normalized_weighted_dist=normalized_weighted_dist)
        return mex_obj
    
    # def plot(self, rule_set, p, r, l):
    #     format_rule_to_atom = self.transfer_rule_to_atom(rule_set, p, r, l) 
    #     rule_highlight = self.obtain_explainable_figure(format_rule_atom, x, data_map_train)
        
    def plot(self, mex_obj, self_row_data=None, all_sequence_data=None, normalized_weighted_dist=None):
        '''
        plot the data in train set or test set:
        trainset: row_data = self.row_train, all_sequence_data = self.all_sequence_data_train
        testset: row_data = self.row_test, all_sequence_data = self.all_sequence_data_test
        '''
        # for the cluster data, we select the sequence in one period' clustering index. 
        rules = mex_obj['classifier'].rules
        for single_rule in rules:
            all_atoms = []
            conj = single_rule.conjunction.predicates
            rule_index = 0
            for pre in conj:
                c = str(pre)
                c = c.split('==')[0]
                c = c.split('_')
                intervals = c[1]
                clusters = c[3]
                all_atoms.append((int(intervals), int(clusters), self.color[rule_index])) 
                rule_index += 1
            print(all_atoms)
            # ! running on the clusterred dataset plot index length include atom color infotmation 
            # [!delete] plot each sequence in the data the sequence clustering and period are both correct 
            # ! plot the period in the data not the sequence 
            plot_index_length = collections.defaultdict(list)
            # plot_weights_dis = self.weighted_dist_logic.detach().cpu().numpy()
            plot_weights_dis = normalized_weighted_dist
            plot_weights_dis = np.reshape(plot_weights_dis, (plot_weights_dis.shape[0], -1, self.n_clustering))
            for sample_index in range(len(plot_weights_dis)):
                for time_zone in range(len(plot_weights_dis[sample_index])):
                    # j is index 
                    # delete the sequence in different time zone
                    # each sequence is closed in left and open in right
                    # time_zone_end = (sequence_index+self.subsequence_length-1)//self.period_length
                    # if time_zone != time_zone_end:
                        # continue
                    clusterings = plot_weights_dis[sample_index][time_zone]
                    sequence_index = time_zone*self.period_length
                    if clusterings.max() == 0: 
                        # if the no clustering activated, this is the filled data
                        continue
                    highlighed_cluster = np.argmax(clusterings)
                    # if (time_zone,highlighed_cluster) in all_atoms:
                    # if one atom in a rule is in the data, then keep this data to plot_index_length, 
                    # ! as long as there is a correct label sequence in the data, then rule learning model can learn it 
                    # ! image plot is more strict than rule, because we only plot the sequence only in one period 
                    for item in all_atoms:
                        if item[0] == time_zone and item[1] == highlighed_cluster:
                            plot_index_length[sample_index].append((time_zone,highlighed_cluster,sequence_index, item[2]))
                        # plot_index_length[i].append((time_zone,highlighed_cluster,j))
            
            # rule also in subqueces data 
            final_plot_index = collections.defaultdict(list) # the key is index sample
            for key in plot_index_length:
                if plot_index_length[key] == []:
                    final_plot_index[key] = []
                    continue
                value = plot_index_length[key]
                all_atoms_seq = [(intervals, clusters) for intervals, clusters, index, _ in value] # the pairs of intervals and clusters in data 
                all_atom_no_color = [(intervals, clusters) for intervals, clusters, index in all_atoms] # the pairs of intervals and clusters in rule
                if set(all_atom_no_color).issubset(set(all_atoms_seq)):
                    final_plot_index[key] = value
                else:
                    final_plot_index[key] = []
            # ! even if the rule is correct, but the image is not presented, then the rule is pass not correct
            all_sequence = []
            sub_sequence = []
            # except the label 
            row_data_label = np.array(self_row_data[self.label_name])
            row_data = np.array(self_row_data.drop(columns=[self.label_name]))
            # running on the original data 
            # todo plot the attention map if the data is image 
            all_mask = []
            for key in final_plot_index:
                single_mask = ['none'] * len(row_data[1])
                value = final_plot_index[key] # ! for negative the value should be nil
                all_sequence.append((row_data[key],row_data_label[key]))
                add_to_sub = []
                for item in value:
                    add_to_sub.append((all_sequence_data[key][item[2]], item[2], item[3]))
                    # change the value 
                    for change_index in range(item[2], item[2]+self.subsequence_length):
                        single_mask[change_index] = item[3][-1]
                sub_sequence.append(add_to_sub)
                all_mask.append(single_mask)
            # add some negative data to the plot
            maximum_negative = 10
            item_index = 0
            succ_neg = 0
            for item in row_data:
                if row_data_label[item_index] == 0:
                    all_sequence.append((item,0))
                    add_to_sub = []
                    sub_sequence.append(add_to_sub)
                    all_mask.append([])
                    succ_neg += 1
                item_index += 1
                if succ_neg > maximum_negative:
                    break
            self.plot_data(all_sequence, sub_sequence, single_rule)
            if self.data_type == 'image':
                self.plot_image(all_sequence, all_mask, single_rule)
        
        return 0
    
    # def plot(self, mex_obj, self_row_data=None, all_sequence_data=None, weighted_dist_logic=None):
    #     '''
    #     plot the data in train set or test set:
    #     trainset: row_data = self.row_train, all_sequence_data = self.all_sequence_data_train
    #     testset: row_data = self.row_test, all_sequence_data = self.all_sequence_data_test
    #     '''
    #     # for the cluster data, we select the sequence in one period' clustering index. 
    #     rules = mex_obj['classifier'].rules
    #     for single_rule in rules:
    #         all_atoms = []
    #         conj = single_rule.conjunction.predicates
    #         rule_index = 0
    #         for pre in conj:
    #             c = str(pre)
    #             c = c.split('==')[0]
    #             c = c.split('_')
    #             intervals = c[1]
    #             clusters = c[3]
    #             all_atoms.append((int(intervals), int(clusters), self.color[rule_index])) 
    #             rule_index += 1
    #         print(all_atoms)
    #         # running on the clusterred dataset
    #         plot_index_length = collections.defaultdict(list)
    #         # plot_weights_dis = self.weighted_dist_logic.detach().cpu().numpy()
    #         plot_weights_dis = weighted_dist_logic
    #         plot_weights_dis = np.reshape(plot_weights_dis, (plot_weights_dis.shape[0], -1, self.n_clustering))
    #         for sample_index in range(len(plot_weights_dis)):
    #             for sequence_index in range(len(plot_weights_dis[sample_index])):
    #                 # j is index 
    #                 time_zone = sequence_index//self.period_length
    #                 # delete the sequence in different time zone
    #                 # each sequence is closed in left and open in right
    #                 time_zone_end = (sequence_index+self.subsequence_length-1)//self.period_length
    #                 if time_zone != time_zone_end:
    #                     continue
    #                 clusterings = plot_weights_dis[sample_index][sequence_index]
    #                 if clusterings.max() == 0: 
    #                     # if the no clustering activated, this is the filled data
    #                     continue
    #                 highlighed_cluster = np.argmax(clusterings)
    #                 # if (time_zone,highlighed_cluster) in all_atoms:
    #                 # if one atom in a rule is in the data, then keep this data to plot_index_length, 
    #                 # ! as long as there is a correct label sequence in the data, then rule learning model can learn it 
    #                 # ! image plot is more strict than rule, because we only plot the sequence only in one period 
    #                 for item in all_atoms:
    #                     if item[0] == time_zone and item[1] == highlighed_cluster:
    #                         plot_index_length[sample_index].append((time_zone,highlighed_cluster,sequence_index, item[2]))
    #                     # plot_index_length[i].append((time_zone,highlighed_cluster,j))
            
    #         # rule also in subqueces data 
    #         final_plot_index = collections.defaultdict(list) # the key is index sample
    #         for key in plot_index_length:
    #             value = plot_index_length[key]
    #             all_atoms_seq = [(intervals, clusters) for intervals, clusters, index, _ in value] # the pairs of intervals and clusters in data 
    #             all_atom_no_color = [(intervals, clusters) for intervals, clusters, index in all_atoms] # the pairs of intervals and clusters in rule
    #             if set(all_atom_no_color).issubset(set(all_atoms_seq)):
    #                 final_plot_index[key] = value
            
    #         all_sequence = []
    #         sub_sequence = []
    #         # except the label 
    #         row_data_label = np.array(self_row_data[self.label_name])
    #         row_data = np.array(self_row_data.drop(columns=[self.label_name]))
    #         # running on the original data 
    #         # todo plot the attention map if the data is image 
    #         all_mask = []
    #         for key in final_plot_index:
    #             single_mask = ['none'] * len(row_data[1])
    #             value = final_plot_index[key] # ! for negative the value should be nil
    #             all_sequence.append((row_data[key],row_data_label[key]))
    #             add_to_sub = []
    #             for item in value:
    #                 add_to_sub.append((all_sequence_data[key][item[2]], item[2], item[3]))
    #                 # change the value 
    #                 for change_index in range(item[2], item[2]+self.subsequence_length):
    #                     single_mask[change_index] = item[3][-1]
    #             sub_sequence.append(add_to_sub)
    #             all_mask.append(single_mask)
    #         # add some negative data to the plot
    #         maximum_negative = 10
    #         item_index = 0
    #         for item in row_data:
    #             if row_data_label[item_index] == 0:
    #                 all_sequence.append((item,0))
    #                 add_to_sub = []
    #                 sub_sequence.append(add_to_sub)
    #                 all_mask.append([])
    #             item_index += 1
    #             if item_index > maximum_negative:
    #                 break
    #         self.plot_data(all_sequence, sub_sequence, single_rule)
    #         if self.data_type == 'image':
    #             self.plot_image(all_sequence, all_mask, single_rule)
        
    #     return 0
    
    def plot_image(self, all_sequence, mask, rule):
        max_image_positive = 10
        plt.cla()
        fig, axs = plt.subplots(max_image_positive, 1, figsize=(28, 28))
        title = str(rule.conjunction)
        precision = rule.precision
        recall = rule.recall
        plt.suptitle(f"{title}, precision {precision}, recall {recall}", fontsize = 24)
        counting = 0
        plt.gca().invert_yaxis()
        for i in range(len(all_sequence)):
            if not (counting < max_image_positive):
                break
            if all_sequence[i][1] == 1 or all_sequence[i][1] == 0:
                cax = axs[counting].imshow(all_sequence[i][0].reshape(self.image_row,self.image_column), cmap='gray')
                # append the logic attention 
                single_mask = mask[i]
                single_mask = np.array(single_mask, dtype=str).reshape(self.image_row,self.image_column)
                for j in range(self.image_row):
                    for k in range(self.image_column):
                        axs[counting].add_patch(plt.Rectangle((k-0.5, j-0.5), 1, 1, color=single_mask[j][k], alpha=0.5))
                axs[counting].set_xticklabels([])
                axs[counting].set_yticklabels([])            
                # axs[counting].set_title(f'Positive {mask[i]}')
                plt.colorbar(cax, ax=axs[counting], orientation='vertical')  
                counting += 1
            else:
                continue
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.savefig(f'{self.result_name}/{time}_{title[0:5]}_image.pdf',bbox_inches='tight')
        return 0 
    
    def plot_data(self, all_data, sub_sequence, rule):
        '''
        all_data: [(data,label)]
        '''
        max_image_positive = 10
        max_image_negative = 10
        # fig, axs = plt.subplots(10, 1, figsize=(10, 5))
        plt.cla()
        fig, axs = plt.subplots(figsize=(10, 4))
        # axs[0].set_ylabel('Value')
        plt.ylabel('Value', fontsize = 24)
        plt.xlabel(f'Time points', fontsize = 24)
        plt.xticks(fontsize=24, rotation=45)
        plt.yticks(fontsize=24)
        title = str(rule.conjunction)
        precision = rule.precision
        recall = rule.recall
        plt.suptitle(f"{title}, precision {precision}, recall {recall}", fontsize = 24)
        labeled_positive = 0
        labeled_negative = 0
        ploted_positive = 0
        ploted_negative = 0
        for i in range(len(all_data)):
            # axs.cla()
            if all_data[i][1] == 1:
                if labeled_positive == 0:
                    label = 'Positive'
                    labeled_positive = 1
                else:
                    label = ''
                spe_color = '#1280ED'
                spe_style = '-'
                ploted_positive += 1
            elif all_data[i][1] == 0 and sub_sequence[i] != []:
                # false positive 
                continue
            elif all_data[i][1] == 0 and sub_sequence[i] == []:
                if labeled_negative == 0:
                    label = 'Negative'
                    labeled_negative = 1
                else:
                    label = ''
                spe_color = '#ED7F12'
                spe_style = '--'
                ploted_negative += 1
            if ploted_positive > max_image_positive and all_data[i][1] == 1:
                continue
            if ploted_negative > max_image_negative and all_data[i][1] == 0:
                continue
            plt.plot(all_data[i][0], spe_color, label=label, linestyle=spe_style)
            for sub_index, sequence_index in enumerate(sub_sequence[i]):
                sequence = sequence_index[0]
                start_index = sequence_index[1]
                plot_x = np.arange(start_index, start_index+self.subsequence_length)
                plt.plot(plot_x, sequence, sequence_index[2],linewidth=4,alpha=0.5)

        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.legend(fontsize=20)
        plt.grid(True)
        plt.savefig(f'{self.result_name}/{time}_{title[0:5]}_all.pdf',bbox_inches='tight')
        print('[Save figure successfully]')
        
    def compare(self):
        pass 
        

def get_all_unsovled_task(binary=True):
    if binary==True:
        string = 'binary'
    else:
        string = 'mul_binaries'
    todo_path = f'UCR/{string}_unsolved/'
    finish_path = f'UCR/{string}_solved/'
    unsolved_list = os.listdir(todo_path)
    finish_list = os.listdir(finish_path)
    todo_list = []
    for i in unsolved_list:
        if i in finish_list:
            continue
        else:
            todo_list.append(i)
    return todo_list

def get_all_task(binary=True):
    if binary==True:
        string = 'binary'
    else:
        string = 'mul_binaries'
    todo_path = f'UCR/{string}_unsolved/'
    finish_path = f'UCR/{string}_solved/'
    unsolved_list = os.listdir(todo_path)
    finish_list = os.listdir(finish_path)
    todo_list = []
    for i in unsolved_list:
        todo_list.append(i)
    return todo_list

def add_task_to_solved_folder(task_name,binary=True):
    if binary==True:
        string = 'binary'
    else:
        string = 'mul_binaries'
    path = f'UCR/{string}_solved/{task_name}'
    if os.path.exists(path) == False:
        os.mkdir(path)
    return 0 


def main(task = 'binary', append_info='',device = ''):
    # father_path = 'rule_learning_original'
    # train_data_path = os.path.join(father_path, 'data/gun_point_train_data_normalized.csv')
    # test_data_path = os.path.join(father_path, 'data/gun_point_test_data_normalized.csv')
    # task_name = 'gun'
    # rule_learning_method = 'ripper'
    # result_name = os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')
    # ts = TimeSeries(rule_learning_method= rule_learning_method ,train_data_path=train_data_path, test_data_path=test_data_path, result_name=result_name, task_name=task_name)
    # ts.run()
    
    
    
    # # SGH tasks 
    # father_path = 'rule_learning_original'
    # train_data_path = os.path.join('SGH_data', 'processed_past24_future12_unit60_min_mrecord10_train.csv')
    # test_data_path = os.path.join('SGH_data', 'processed_past24_future12_unit60_min_mrecord10_test.csv')
    # task_name = 'sgh'
    # rule_learning_method = 'ripper'
    # middle_data_path = os.path.join(father_path, 'data', f'{task_name}_')
    # result_name = os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')
    # ts = TimeSeriesNan(rule_learning_method= rule_learning_method ,train_data_path=train_data_path, test_data_path=test_data_path, result_name=result_name, task_name = task_name, middle_data_path=middle_data_path,past_windows_time=24, unit = 1, low=4, high=11)
    # ts.run()
    
    
    
    ###setting for GUN ###
    # data_name = 'rule_learning_original/data/gun'
    # train_data_path = os.path.join(data_name, 'gun_point_train_data.csv')
    # test_data_path = os.path.join(data_name, 'gun_point_test_data.csv')
    # task_name = 'gun'
    # rule_learning_method = 'ripper'
    # result_name = os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')    
    # ts = TimeZoneSeries(rule_learning_method= rule_learning_method ,train_data_path=train_data_path, test_data_path=test_data_path, result_name=result_name, task_name = task_name, father_path=father_path, unit = 1, low=None, high=None, clustering_method='tsDWTmeans', max_plot = 50)
    # ts.only_plot_rules()
    
    ### setting for SGH data ###
    # data_name = 'SGH_data/mulevent'
    # train_data_path = os.path.join(data_name, 'new_ins_glu_past24_target4_future_None_unit60_min_mrecord4_group_all_year_2018_abnormal_train.csvfull')
    # test_data_path = os.path.join(data_name, 'new_ins_glu_past24_target4_future_None_unit60_min_mrecord4_group_all_year_2019_abnormal_train.csvfull')
    # task_name = 'sgh_4point_mulevent_1819_abn'
    # # rule_learning_method = 'ripper'
    # rule_learning_method = 'DFORL'
    # result_name = os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')    
    # ts = TimeZoneSeries(rule_learning_method= rule_learning_method ,train_data_path=train_data_path, test_data_path=test_data_path, result_name=result_name, task_name = task_name, father_path=father_path, unit = 1, low=4, high=10, clustering_method='tsDWTmeans', max_plot=50, max_inter=100, n_clustering=20)
    # # ts.run()
    # ts.only_plot_rules()
    
    ## setting for NEW SGH data generated from xy version ###
    # data_name = 'SGH_data/mulevent_new'
    # train_data_path = os.path.join(data_name, 'new_ins_glu_past24_target6_future_6_unit60_min_mrecord4_group_all_year_2018_high_xnew_train.csvfull')
    # test_data_path = os.path.join(data_name, 'new_ins_glu_past24_target6_future_6_unit60_min_mrecord4_group_all_year_2019_high_xnew_train.csvfull')
    # task_name = 'newxy_sgh_1819_high_6'
    # # rule_learning_method = 'ripper'
    # rule_learning_method = 'DFORL'
    # result_name = os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')    
    # ts = TimeZoneSeries(rule_learning_method= rule_learning_method ,train_data_path=train_data_path, test_data_path=test_data_path, result_name=result_name, task_name = task_name, father_path=father_path, unit = 1, low=70, high=140, clustering_method='tsDWTmeans', max_plot=50, max_inter=100, n_clustering=5)
    # ts.run()
    # ts.only_plot_rules()
    
    ### setting for mimic data ###
    # data_name = 'mimic_data/mulevent'
    # train_data_path = os.path.join(data_name, 'processed_ins_glu_past24_target1_future_24_unit60_min_mrecord5_group_all_train.csv')
    # test_data_path = os.path.join(data_name, 'processed_ins_glu_past24_target1_future_24_unit60_min_mrecord5_group_all_test.csv')
    # task_name = 'mimic_mulevent'
    # rule_learning_method = 'ripper'
    # result_name = os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')    
    # # ts = TimeZoneSeries(rule_learning_method= rule_learning_method ,train_data_path=train_data_path, test_data_path=test_data_path, result_name=result_name, task_name = task_name, father_path=father_path, unit = 1, low=70, high=180, clustering_method='tsDWTmeans', max_plot=50)
    
    # or mimic data ###
    # data_name = 'mimic_data/mulevent'
    # train_data_path = os.path.join(data_name, 'processed_ins_glu_past24_target1_future_24_unit60_min_mrecord5_group_all_train.csv')
    # test_data_path = os.path.join(data_name, 'processed_ins_glu_past24_target1_future_24_unit60_min_mrecord5_group_all_test.csv')
    # task_name = 'mimic_mulevent'
    # # rule_learning_method = 'ripper'
    # rule_learning_method = 'DFORL'
    # result_name = os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')    
    # ts = TimeZoneSeries(rule_learning_method= rule_learning_method ,train_data_path=train_data_path, test_data_path=test_data_path, result_name=result_name, task_name = task_name, father_path=father_path, unit = 1, low=4, high=10, clustering_method='tsDWTmeans', max_plot=50, max_inter=100)
    # ts.run()
    
    # generate random seeds 
    # random_seed = [1,2,3,4,5,100,200,300,400,1231]
    # for i in random_seed:
    #     torch.manual_seed(i)
    #     ### setting for UCR coffee ###
    #     archive_name = 'Coffee'
    #     data_name = f'UCR/{archive_name}/'
    #     train_data_path = os.path.join(data_name, f'{archive_name}_train.csv')
    #     test_data_path = os.path.join(data_name, f'{archive_name}_test.csv')
    #     task_name = f'ucr_{archive_name}'
    #     # rule_learning_method = 'ripper'
    #     rule_learning_method = 'DFORL_endtoend'
    #     result_name = os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')    
    #     ts = EndToEndTimeSeries(rule_learning_method= rule_learning_method ,train_data_path=train_data_path, test_data_path=test_data_path, result_name=result_name, task_name = task_name, father_path=father_path, unit = 1, low=None, high=None, max_plot=50, min_precision=0.8, min_recall=0.8, max_inter=10, n_clustering=specs.n_clusters, model_type='time_period')
    #     ts.run_loop()
    # ts.only_plot_rules()


    def demo(device, append_info):
        # demo data 
        random_seed = [1,2,3,4,5,100,200,300,400,1231]
        random_seed = [1]
        for single_seed in random_seed:
            torch.manual_seed(single_seed)
            np.random.seed(single_seed)
            ### setting for UCR coffee ###
            archive_name = 'demo_threepatten'
            test_flag = True
            data_name = f'UCR/{archive_name}/'
            train_data_path = os.path.join(data_name, f'{archive_name}_train.csv')
            test_data_path = os.path.join(data_name, f'{archive_name}_test.csv')
            task_name = f'ucr_{archive_name}'
            # rule_learning_method = 'ripper'
            rule_learning_method = 'DFORL_endtoend'
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result_name = os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')    
            ts = EndToEndTimeSeries(rule_learning_method= rule_learning_method ,train_data_path=train_data_path, test_data_path=test_data_path, result_name=result_name, task_name = task_name, father_path=father_path, unit = 1, low=None, high=None, max_plot=50, min_precision=0.8, min_recall=0.8, max_inter=10, n_clustering=specs.n_clusters, model_type='time_period', device=device, test_flag=test_flag)
            ts.run_loop()
        return 0 
    

    def UCR_binary(device, append_info):
        # UCR binary data 
        random_seed = [1,100,1231]
        for single_seed in random_seed:
            # torch.manual_seed(single_seed)
            # np.random.seed(single_seed)
            # todo_task = get_all_unsovled_task()
            # for task in todo_task:
            todo_task = get_all_task()
            # focused_task = ['Distal','Earth','ECG','Gun_Point','Ham','Hand']
            focused_task = ['Italy']
            for task in todo_task:
                for subtask in focused_task:
                    if subtask in task:
                        continue_flag = 0
                        break
                    else:
                        continue_flag = 1
                if continue_flag == 1:
                    continue
                # line for debug
                print(colored(task, 'red'), colored(single_seed,'green'))
                time.sleep(3)
                ### setting for UCR coffee ###
                start_time = datetime.now()
                test_flag = True
                data_name = f'UCR/binary_unsolved/{task}/'
                train_data_path = os.path.join(data_name, f'{task}_train.csv')
                test_data_path = os.path.join(data_name, f'{task}_test.csv')
                task_name = f'ucr_{task}'
                # rule_learning_method = 'ripper'
                rule_learning_method = 'DFORL_endtoend'
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result_name = os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')    
                ts = EndToEndTimeSeries(rule_learning_method= rule_learning_method ,train_data_path=train_data_path, test_data_path=test_data_path, result_name=result_name, task_name = task_name, father_path=father_path, unit = 1, low=None, high=None, max_plot=50, min_precision=0.1, min_recall=0.8, max_inter=10, n_clustering=specs.n_clusters, model_type='time_period', device=device, test_flag=test_flag)
                ts.run_loop()
                end_time = datetime.now()
                running_time = end_time - start_time
                with open('time_deeprl.md','a+') as f:
                    print('Task:',task, 'RTime:',running_time, file=f)
                add_task_to_solved_folder(task)
    
    def UCR_mul(device, append_info):
        # UCR multivariable data 
        random_seed = [1,1231,100]
        for single_seed in random_seed:
            torch.manual_seed(single_seed)
            np.random.seed(single_seed)
            # todo_task = get_all_unsovled_task(binary=False)
            todo_task = get_all_task(binary=False)
            focused_task = ['WordSynonyms_9']
            # focused_task = ['Lightning7_1','Lightning7_2','Lightning7_3','Lightning7_4']
            # focused_task = ['OliveOil']
            for task in todo_task:
                for subtask in focused_task:
                    if subtask in task:
                        continue_flag = 0
                        break
                    else:
                        continue_flag = 1
                if continue_flag == 1:
                    continue
            # for task in todo_task:
                task_name = task.split('_')[0]
                task_label = task.split('_')[1]
                print(colored(task, 'red'), colored(single_seed,'green'))
                time.sleep(3)
                ### setting for UCR coffee ###
                test_flag = True
                data_name = f'UCR/mul_binaries_unsolved/{task}/'
                train_data_path = os.path.join(data_name, f'{task_name}_train.csv')
                test_data_path = os.path.join(data_name, f'{task_name}_test.csv')
                task_name = f'ucr_{task_name}'
                rule_learning_method = f'DFORL_endtoend_{task_label}'
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result_name = os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')    
                ts = EndToEndTimeSeries(rule_learning_method= rule_learning_method ,train_data_path=train_data_path, test_data_path=test_data_path, result_name=result_name, task_name = task_name, father_path=father_path, unit = 1, low=None, high=None, max_plot=50, min_precision=0.1, min_recall=0.8, max_inter=10, n_clustering=specs.n_clusters, model_type='time_period', device=device, test_flag=test_flag)
                ts.run_loop()
                add_task_to_solved_folder(task,binary=False)
    
    def image(device, append_info):
        # demo data 
        # random_seed = [1,100,1231]
        random_seed = [1]
        for single_seed in random_seed:
            torch.manual_seed(single_seed)
            np.random.seed(single_seed)
            ### setting for UCR coffee ###
            archive_name = 'MNIST'
            print(archive_name)
            print(append_info)
            time.sleep(3)
            test_flag = True
            data_name = f'{archive_name}'
            train_data_path = os.path.join(data_name, f'{archive_name}_train_{append_info}.csv')
            test_data_path = os.path.join(data_name, f'{archive_name}_test_{append_info}.csv')
            task_name = f'ucr_{archive_name}_{append_info}'
            # rule_learning_method = 'ripper'
            rule_learning_method = 'DFORL_endtoend'
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result_name = os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')    
            ts = EndToEndTimeSeries(rule_learning_method= rule_learning_method ,train_data_path=train_data_path, test_data_path=test_data_path, result_name=result_name, task_name = task_name, father_path=father_path, unit = 1, low=None, high=None, max_plot=50, min_precision=0.1, min_recall=0.8, max_inter=10, n_clustering=specs.n_clusters, model_type='time_period', device=device, test_flag=test_flag, data_type='image')
            ts.run_loop()
    
    # def image_mul(device, append_info):
    #     # demo data 
    #     random_seed = [1,100,1231]
    #     for single_seed in random_seed:
    #         torch.manual_seed(single_seed)
    #         np.random.seed(single_seed)
    #         ### setting for UCR coffee ###
    #         archive_name = 'MNIST'
    #         print(archive_name)
    #         time.sleep(3)
    #         test_flag = True
    #         data_name = f'{archive_name}'
    #         train_data_path = os.path.join(data_name, f'{archive_name}_train_{append_info}.csv')
    #         test_data_path = os.path.join(data_name, f'{archive_name}_test_{append_info}.csv')
    #         task_name = f'ucr_{archive_name}_{append_info}'
    #         # rule_learning_method = 'ripper'
    #         rule_learning_method = 'DFORL_endtoend'
    #         current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         result_name = os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')    
    #         ts = EndToEndTimeSeries(rule_learning_method= rule_learning_method ,train_data_path=train_data_path, test_data_path=test_data_path, result_name=result_name, task_name = task_name, father_path=father_path, unit = 1, low=None, high=None, max_plot=50, min_precision=0.8, min_recall=0.8, max_inter=10, n_clustering=specs.n_clusters, model_type='time_period', device=device, test_flag=test_flag, data_type='image')
    #         ts.run_loop()
    
    
    specs.data = task
    father_path = 'rule_learning_original'  
    specs.main()
    print(colored(f'Device {device} Task {task}', 'red'))
    if task == 'demo':
        demo(device, append_info)
    elif task == 'UCR_binary':
        UCR_binary(device, append_info)
    elif task == 'UCR_mul':
        UCR_mul(device, append_info)
    elif  task == 'image':
        image(device, append_info)
    else:
        raise ValueError('Task not found')
    
    # setting for multivariable data     
    # # todo_task = get_all_unsovled_task(binary=False)
    # todo_task = get_all_task(binary=False)
    # focused_task = ['CBF','FaceFour','Lighting2','Lighting7','OSULeaf','Trace','WordsSynonyms','OliveOil','StarLightCurves']
    # for task in todo_task:
    #     for subtask in focused_task:
    #         if subtask in task:
    #             continue_flag = 0
    #             break
    #         else:
    #             continue_flag = 1
    #     if continue_flag == 1:
    #         continue
    #     task_name = task.split('_')[0]
    #     task_label = task.split('_')[1]
    #     print(task)
    #     data_name = f'UCR/mul_binaries_unsolved/{task}/'
    #     train_data_path = os.path.join(data_name, f'{task_name}_train.csv')
    #     test_data_path = os.path.join(data_name, f'{task_name}_test.csv')
    #     task_name = f'ucr_{task_name}'
    #     # rule_learning_method = 'ripper'
    #     rule_learning_method = 'DFORL_'+task_label
    #     result_name = os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')    
    #     ts = TimeZoneSeries(rule_learning_method= rule_learning_method ,train_data_path=train_data_path, test_data_path=test_data_path, result_name=result_name, task_name = task_name, father_path=father_path, unit = 1, low=None, high=None, clustering_method='tsDWTmeans', max_plot=10, max_inter=10,multi_class_label=data_name, n_clustering=5)
        
    #     # print all parameters in a file
    #     if os.path.exists(os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')) == False:
    #         os.mkdir(os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}'))
    #     with open(os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}', 'parameters.md'), 'a+') as f:
    #         f.write(f'**{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}**\n')
    #         print(str(ts.__dict__), file=f)
    #     ts.run()
    #     add_task_to_solved_folder(task,binary=False)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--task', type=str, default='b')
    parser.add_argument('-a','--append_data_info', type=str, default='')
    parser.add_argument('-d','--device', type=str, default='cuda:0')
    args = parser.parse_args()
    if args.task == 'm':
        main('UCR_mul', append_info=args.append_data_info, device = args.device)
    elif args.task == 'b':
        main('UCR_binary', append_info=args.append_data_info, device = args.device)
    elif args.task == 'd':
        main('demo', append_info=args.append_data_info, device = args.device)
    elif args.task == 'i':
        main('image', append_info=args.append_data_info, device = args.device)
    elif args.task == 'im':
        append_info = ['p1_n0']
        # append_info = ['p1nrest','p2nrest','p3nrest','p4nrest','p5nrest']
        for i in append_info:
            main('image', append_info=i, device = args.device)
