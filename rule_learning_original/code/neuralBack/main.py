import torch 
import numpy as np 
import torch.nn as nn
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
import sklearn.metrics
import matplotlib.pyplot as plt
import sys 
sys.path.append('rule_learning_original/code/neuralBack/')
from nerual_kan import *
from dforl_torch import *
import pickle
from metrics_checker import *
from datetime import datetime 
import tqdm 
# valid data 
class mymodel(nn.Module):
    def __init__(self,number_features):
        super(mymodel, self).__init__()
        self.fc1 = nn.Linear(number_features, 100, bias=False)
        # self.batchnorm1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 1, bias=False)
        self.activation = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        # x = self.batchnorm1(x)
        x = self.fc2(x)
        # x = self.dropout(x)
        x= self.sigmoid(x)
        # x = self.batchnorm2(x)
        # x = self.fc3(x)
        # x = self.activation(x)
        return x

class rule_nn_model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nn1 = mymodel()
        self.nn2 = mymodel()
        self.rule = RuleLayer(in_size=2,rule_number=2)
    def forward(self, x):
        x1 = self.nn1(x)
        x2 = self.nn2(x)
        rule_input = torch.cat([x1, x2], dim=1)
        x = self.rule(rule_input)
        return x
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class RunModel():
    def __init__(self, train:pd.DataFrame=None, test:pd.DataFrame=None, input_file_name = '',remove_low_precision=0.8, learn_rate = 0.1,  epoch = 1000, output_file_name = 'out', minimum_precision = 0.8, minimum_recall =0.8, maximum_iteration = 10,early_stopping = False, batch_size = 1024, model_type='single', t_relation = '', task_name='',t_arity=2,kg_problem=False, device = 'cuda:0'):
        '''
        model_type: 'single' or 'deep' or 'resident'
        '''
        self.train = train
        self.test = test
        self.input_file_name = input_file_name
        self.remove_low_precision = remove_low_precision
        self.learn_rate = learn_rate
        self.epoch = epoch
        self.output_file_name = output_file_name
        self.minimum_precision = minimum_precision
        self.minimum_recall = minimum_recall
        self.maximum_iteration = maximum_iteration
        self.early_stop = early_stopping
        self.batch_size = batch_size
        self.mode_type = model_type # 'single' or 'deep' or  'resident'
        self.t_relation = t_relation
        self.task = task_name
        self.t_arity = t_arity
        self.kg_problem = kg_problem
        self.device = device
        self.time_stamp = datetime.now()


            
    def single_run(self):
        self.early_stopper = EarlyStopper(patience=10, min_delta=0.0)
        if self.train is None or self.test is None:
            csv_data = pd.read_csv(f'{self.input_file_name}')

            # positive_data = csv_data[csv_data['label'] == 1]
            # negative_data = csv_data[csv_data['label'] == 0]
            # fist_atom = positive_data['feature0_1'].astype(int)
            # second_atom = positive_data['feature2_3'].astype(int)
            # assert or_atoms.min() == 1

            number_instances = csv_data.shape[0]
            number_features = csv_data.shape[1] - 1
            print(number_instances)
            ## Use pytorch to train the data
            # make trainable data with torch 
            
            # torch.manual_seed(9504001738191942965)
            # train_data = csv_data[:int(0.8*number_instances)]   
            # test_data = csv_data[int(0.8*number_instances):]
            
            train_data = csv_data
            test_data = None
        else:
            train_data = self.train
            test_data = self.test
            number_instances = train_data.shape[0]
            number_features = train_data.shape[1] - 1
            

        train_target = torch.tensor(train_data['label'].values.astype(np.float32)).reshape(-1,1)
        train_x = torch.tensor(train_data.drop('label', axis = 1).values.astype(np.float32))
        train_tensor = torch.utils.data.TensorDataset(train_x, train_target) 
        train_loader = torch.utils.data.DataLoader(dataset = train_tensor, batch_size = self.batch_size, shuffle = True, pin_memory=True, num_workers= 1)

        if type(test_data) is not type(None):
            test_target = torch.tensor(test_data['label'].values.astype(np.float32)).reshape(-1,1)
            test_x = torch.tensor(test_data.drop('label', axis = 1).values.astype(np.float32))
            test_tensor = torch.utils.data.TensorDataset(test_x, test_target)
            test_loader = torch.utils.data.DataLoader(dataset = test_tensor, batch_size = self.batch_size, shuffle = True,pin_memory=True, num_workers= 12)
        
            number_positive_test = test_data['label']
            number_positive_test = number_positive_test[number_positive_test == 1].shape[0]
            number_negative_test = test_data['label']
            number_negative_test = number_negative_test[number_negative_test == 0].shape[0]
            print('number of positive test instances: ', number_positive_test) 
            print('number of negative test instances: ', number_negative_test)
            print(number_negative_test/(number_positive_test + number_negative_test))
        else:
            test_loader = []
            number_positive_test = train_data['label']
            number_positive_test = number_positive_test[number_positive_test == 1].shape[0]
            number_negative_test = train_data['label']
            number_negative_test = number_negative_test[number_negative_test == 0].shape[0]
            print('number of positive test instances: ', number_positive_test) 
            print('number of negative test instances: ', number_negative_test)
            print(number_negative_test/(number_positive_test + number_negative_test))

        # model define 
        if self.mode_type == 'basic':
            model = RuleLayer(in_size=number_features, rule_number=2, output_rule_file=self.output_file_name,t_arity=self.t_arity,device=self.device, time_stamp=self.time_stamp).to(self.device)
        elif self.mode_type == 'deep':
            model = DeepRuleLayer(in_size=number_features, rule_number=2, output_rule_file=self.output_file_name, t_relation=self.t_relation, task=self.task,t_arity=self.t_arity, device=self.device,time_stamp=self.time_stamp).to(self.device)
        elif self.mode_type == 'resident':
            model = DeepResidualLogic(in_size=number_features, rule_number=2, output_rule_file=self.output_file_name,t_arity=self.t_arity, residual_layer=1, device=self.device,time_stamp=self.time_stamp).to(self.device)
        elif self.mode_type == 'normal':
            model = DNN_normal(number_features).to(self.device)
        elif self.mode_type == 'deep_2' or self.mode_type == 'single':
            model = DeepRuleLayer_v2(in_size=number_features, rule_number=2, output_rule_file=self.output_file_name, t_relation=self.t_relation, task=self.task,t_arity=self.t_arity, device=self.device,time_stamp=self.time_stamp).to(self.device)
        else:
            raise ValueError('model type is not defined')
        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learn_rate) # for synthetic data
        writer = SummaryWriter(f'demo/{self.time_stamp}/')
        # print traintable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
        

        for i in range(self.epoch):
            model.train()
            training_loss = 0
            correct = 0
            acc_train = 0
            train_bar = tqdm.tqdm(total=len(train_loader)+len(test_loader), ascii=True, desc='Epoch {}'.format(i))
            for batch_idx, (data, target) in (enumerate(train_loader)):
                data = data.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss += 0.5*model.similiarity()
                    # + 0.01*model.similarity_with_learned_rules()
                loss.backward()
                optimizer.step()
                binary_output = torch.where(output > 0.5, 1, 0)
                correct += torch.sum(torch.eq(target, binary_output)).item()
                training_loss += loss.item()
                train_bar.update(1)
            training_loss = training_loss/len(train_loader.dataset)
            acc_train = correct/len(train_loader.dataset)
            train_bar.set_description('Epoch {}, train loss: {}, train acc {}'.format(i, training_loss, acc_train))
            
            
            if type(test_data) is not  type(None):
                # test on test data
                model.eval()
                correct = 0
                test_loss_all = 0
                for batch_idx, (data, target) in enumerate(test_loader):
                    train_bar.update(1)
                    data = data.to(self.device)
                    target = target.to(self.device)
                    output = model(data)
                    test_loss = criterion(output, target)
                    model.interpret_weights_computation()
                    test_loss += 0.5*model.similiarity() + 0.01*model.similarity_with_learned_rules()
                    binary_output = torch.where(output > 0.5, 1, 0)
                    correct += torch.sum(torch.eq(target, binary_output)).item()
                    test_loss_all += test_loss.item()
                test_loss = test_loss_all/len(test_loader.dataset)
                acc_test = correct/len(test_loader.dataset)
                print('train loss: {}, test loss {}, test acc {}'.format(training_loss, test_loss, acc_test)) 
                # ...log the running loss
            
            if type(test_data) is not type(None):
                writer.add_scalars('loss', {'train': training_loss, 'test': test_loss}, i)
                writer.add_scalar('accuracy', acc_test, i)
                if self.early_stopper.early_stop(test_loss) and self.early_stop:             
                    break
                if acc_test == 1:
                    print('Early Stopping because of perfect accuracy on test data')
                    break
            else:
                writer.add_scalar('loss', training_loss, i)
                writer.add_scalar('accuracy', acc_train, i)
                if self.early_stopper.early_stop(training_loss) and self.early_stop:
                    break
                if acc_train > 0.99:
                    print('Early Stopping because of perfect accuracy on training data')
                    break
            # record parameters in board 
            if i % 50 == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(name, param, i)



        if type(test_data) is type(None):
            test_x = train_x
            y_test = train_target
            test_data = train_data
        else:
            y_test = test_target
            pass 
        
        predict = model(test_x.to(self.device))
        predict = torch.where(predict > 0.5, 1, 0)
        # y_test = test_target.to(self.device)
        # print(predict.cpu().tolist(), y_test.cpu().tolist())
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test.cpu(), predict.cpu()).ravel()
        print('tn', tn, 'fp' ,fp, 'fn', fn, 'tp', tp)
        print(f'From NN: acc_test:{(tp+tn)/(tn+fp+fn+tp)}', f'precision:{tp/(tp+fp)}', f'recall:{tp/(tp+fn)}', f'f1:{2*tp/(2*tp+fp+fn)}')

        # print trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

        rule = model.interpret(train_data)
        
        metric_obj = model.check_metric(test_data, rule, remove_low_precision=self.remove_low_precision, update_parameters=True, single_check=True)
        metric, rule, classifier = metric_obj['metrics'], metric_obj['rule_set'], metric_obj['classifier']
        
        # save rule model 
        with open('rule_model.pkl', 'wb') as f:
            pickle.dump(rule, f)
            f.close()
            
        if self.mode_type == 'deep':
            invented_predicates = model.get_invented_predicate_v2()
            # best_threshold, removed_or_list = model.check_best_rules_with_threshold()
            # invented_predicates = model.interpret_invented_predicate(best_threshold,removed_or_list)
            # model.check_invented_precision_recall()
        return rule, metric, model, test_data, classifier
    
    def check_recall_from_KG(self, classifier):
        metrics = CheckMetrics(t_relation=self.t_relation, task_name=self.task, classifier=classifier,logic_path=f'rule_learning_original/code/neuralBack/res/{self.task}.md', t_arity=self.t_arity)
        KG_recall = metrics.check_recall_of_logic_program()['recall']
        return KG_recall
    
    def run(self):
        rule, metric,_, test_data, classifier = self.single_run()
        KG_recall = 0
        # ! compute the recall of the logic program from KG not update KG recall because of predicate invention 
        if self.kg_problem==True:
            KG_recall = self.check_recall_from_KG(classifier)
        recall = max(KG_recall, metric['rule_set_recall'])
        ini_inter = 0
        while (metric['rule_set_precision'] < self.minimum_precision or recall < self.minimum_recall) and ini_inter < self.maximum_iteration:
            new_rule, _, model, _, classifier= self.single_run()
            new_rule_conj = new_rule.conjunctions
            for conj in new_rule_conj:
                rule.add_conjunction(conj)
            metric_obj = model.check_metric(test_data, rule, remove_low_precision=self.remove_low_precision)
            metric, rule, classifier = metric_obj['metrics'], metric_obj['rule_set'], metric_obj['classifier']
            # ! compute the recall of the logic program from KG not update KG recall because of predicate invention 
            if self.kg_problem==True:
                KG_recall = self.check_recall_from_KG(classifier)
            recall = max(KG_recall, metric['rule_set_recall'])
            ini_inter += 1
        # save rule 
        with open(f'{self.output_file_name}', 'a+') as f:
            print(torch.seed(),file=f)
            f.close()
        with open(f'{self.output_file_name}classifier.pkl', 'wb') as f:
            pickle.dump(classifier, f)
            f.close()
        return rule, metric, classifier

        
        
        
if __name__=='__main__':
    feature_number = 10000
    # exp = RunModel(minimum_precision=1,minimum_recall=0.8, maximum_iteration=10, input_file_name=f'rule_learning_original/code/neuralBack/data/boolean_data_{feature_number}_features.csv', remove_low_precision=0.9, learn_rate = 0.1, epoch=200, output_file_name=f'rule_learning_original/code/neuralBack/res/{feature_number}_features.md', early_stopping=True, batch_size = 10240, model_type='deep_2')
    
    task_name = 'gp'
    head_predicate_name = 'gp'
    variable_length = '1'
    t_arity= 2
    #! Train the model and obtain the rule 
    exp = RunModel(minimum_precision=1,minimum_recall=1, maximum_iteration=10, input_file_name=f'rule_learning_original/code/DFORL/{task_name}/data/{head_predicate_name}/data{variable_length}.csv', remove_low_precision=0.9, learn_rate =1, epoch=2000, output_file_name=f'rule_learning_original/code/neuralBack/res/{task_name}.md', early_stopping=True, batch_size = 100000, model_type='deep',t_relation = head_predicate_name, task_name=task_name, t_arity = t_arity,kg_problem=True, device='cuda:1')
    exp.run()

    
    # ! compute the recall of the logic program from KG
    result_file = f'rule_learning_original/code/neuralBack/res/{task_name}.md'
    with open(f'{result_file}classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
        f.close()
    KG_recall = exp.check_recall_from_KG(classifier)
    # metrics = CheckMetrics(t_relation=task_name, task_name=task_name, classifier=classifier,logic_path=f'rule_learning_original/code/neuralBack/res/{task_name}.md')
    # metrics.check_recall_of_logic_program()
    
    