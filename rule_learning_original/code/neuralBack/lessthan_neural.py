import torch 
import torcheval.metrics 
import torch.nn as nn
from torch.autograd.function  import InplaceFunction
import argparse
import pandas as pd
import numpy as np
import collections
import torch.utils 
import sys, os
sys.path.append('/home/gaokun/kk/')
from aix360_kun.aix360.algorithms.rule_induction.trxf.core.dnf_ruleset import DnfRuleSet
from aix360_kun.aix360.algorithms.rule_induction.trxf.core.conjunction import Conjunction
from aix360_kun.aix360.algorithms.rule_induction.trxf.core.predicate import Predicate, Relation
from torch.utils.tensorboard import SummaryWriter
from rule_learning_original.code.pykan.kan import *
## set the argument


class RuleLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    
    def __init__(self, in_size, rule_number, default_alpha = 10, device='cuda:0'):
        super().__init__()
        self.size_in = in_size
        self.rule_number = rule_number
        self.alpha = default_alpha
        weights = torch.Tensor(self.rule_number, self.size_in)
        self.weights = nn.Parameter(weights, requires_grad=True)  # nn.Parameter is a Tensor that's a module parameter.
        nn.init.normal_(self.weights, mean=0.5, std=0.1) # weight init
        self.device = device

    # def sigmoid_like(self, x):
    #     y = torch.divide(1,(1+torch.exp(-1 * self.alpha *(x))))  
    #     return y 
    
    def fuzzy_or(self, x):   
        neg_inputs = 1 - x
        pro_neg_input = torch.prod(neg_inputs, 1)
        predict_value = 1 - pro_neg_input
        predict_value = torch.reshape(predict_value, [-1, 1])
        return predict_value
    
    def forward(self, x):
        self.interpretable_rul_weights = torch.nn.functional.softmax(self.weights, dim=1)
        clipped_weights = torch.clamp(self.weights, 0, 1)
        w_times_x= torch.mm(x, self.interpretable_rul_weights.t())
        biased = w_times_x - 0.5
        activated_values =  2 * nn.functional.relu(biased)
        rule_satisfy = self.fuzzy_or(activated_values)
        return rule_satisfy # let the rule satisfy be 1 always 
    

class BinarizeLinear(nn.Module):

    def __init__(self, input, output,*kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)
        self.weight = nn.Parameter(nn.init.normal_(torch.empty(input,output), mean=0, std=1), requires_grad=True)

    def forward(self, input):
        self.weight_b=binarized(self.weight)
        out = torch.matmul(input,self.weight_b)
        return out



class learn_minus(nn.Module):
    def __init__(self, number_features,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.number_features = number_features
        self.ps_1 = nn.Parameter(nn.init.normal_(torch.empty(self.number_features,1), mean=1, std=1), requires_grad=True)
        self.ps_2 = nn.Parameter(nn.init.normal_(torch.empty(self.number_features,1),mean=1, std=1), requires_grad=True)
        
        self.binary = BinarizeLinear(2, 1)
        
    def forward(self,x):
        self.only_positive_1 = torch.clamp(self.ps_1, min=0)
        self.only_positive_2 = torch.clamp(self.ps_2, min=0) 
        x1 = torch.matmul(x, self.only_positive_1)
        x2 = torch.matmul(x, self.only_positive_2)
        x = torch.cat([x1,x2], dim=1)
        x = self.binary(x)
        return x
    
    def print_parameters(self):
        for name, param in self.named_parameters():
            print(name, param)
        print(self.only_positive_1)
        print(self.only_positive_2)
        print(self.binary.weight_b)
        return 0
    
    def binary_loss(self):
        loss = 0 
        for name, param in self.named_parameters():
            if 'ps' in name:
                loss += torch.sum(torch.abs(param))
        return loss

def train_model(model, trainloader, testloader, epoch,lr=0.001, max_early_stop = 10):
    writer = SummaryWriter('demo')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stop = 0
    best_loss = float('inf')
    for i in range(epoch):
        if early_stop > max_early_stop:
            break
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate (trainloader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            binary_loss = model.binary_loss()
            loss += 0.1*binary_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                test_loss += criterion(output, target).item()
                metric = torcheval.metrics.BinaryAccuracy(threshold=0.5)
                output = torch.reshape(output, [-1])
                target = torch.reshape(target, [-1])
                metric.update(output, target)
                acc = metric.compute()
                correct += acc*len(target)
        test_loss /= len(testloader.dataset)
        acc = 100. * correct / len(testloader.dataset)
        if total_loss < best_loss:
            best_loss = total_loss
            early_stop = 0
        else:
            early_stop += 1
        print('Train Loss: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, Epoch: {}/{}'.format(total_loss , test_loss, acc, i, epoch))
        writer.add_scalars('loss_demo', {'train': total_loss, 'test': test_loss}, i)
        writer.add_scalar('accuracy_test', acc, i)
    model.print_parameters()
    return 0

class Binarize(InplaceFunction):

    def forward(ctx,input,quant_mode='customized',allow_scale=False,inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()      

        scale= output.abs().max() if allow_scale else 1

        if quant_mode=='det':
            return output.div(scale).sign().mul(scale)
        elif quant_mode == 'customized':
            clamp_index = output.abs() < 1
            output[clamp_index] = 0
            output.clamp_(-1,1)
            return output
        else:
            output = output / scale 
            linear_output = (1+output)/2
            output = (torch.clamp(linear_output,0,1)*2-1)
            return output
            # return output.div_(scale).add_(1).div_(2).add_(torch.rand(output.size()).add_(-0.5)).clamp_(0,1).round_().mul_(2).add_(-1).mul_(scale)
        
        
    def backward(ctx,grad_output):
        #STE 
        grad_input=grad_output
        return grad_input,None,None,None


def binarized(input,quant_mode='det',allow_scale=False):
        return Binarize.apply(input,quant_mode,allow_scale)  

class RuleSet():
    def __init__(self):
        self.disjunctions = []
    def updates(self,conjunctions):
        conjunction_set = set(conjunctions)
        if conjunction_set not in (self.disjunctions):
            self.disjunctions.append(conjunction_set)
    
class NeuralPredicate(nn.Module):
    '''
    The input node should be equal with the number of features. 
    The values for the corresponding features indicate the occurring time of the feature. 
    If the value is 0, then the feature is not occur in a continues data 
    '''
    def __init__(self, number_features=3, number_predicate=2,number_rules=1):
        super(NeuralPredicate, self).__init__()
        parser = self.argument()
        self.args = parser.parse_args()
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        torch.manual_seed(self.args.seed)
            
        self.number_features =  number_features
        self.number_predicates = number_predicate
        self.number_rules = number_rules
        self.kan = KAN(width=[self.number_features,self.number_predicates], grid=5,k=3,seed=0)
        self.rule = RuleLayer(self.number_predicates, self.number_rules)
        self.feature_name = {}
        for i in range(self.number_features):
            self.feature_name[i] = f'feature_{i}'
        
        # self.criterion = torch.nn.MSELoss()
        # self.optimizer = torch.optim.Adam([
        #         {'params': self.rule.parameters()},
        #     ], lr=0.0001, weight_decay=0.9)
    
    def  argument(self):
        parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                            help='input batch size for training (default: 256)')
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=50, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                            help='learning rate (default: 0.001)')
        parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                            help='SGD momentum (default: 0.5)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=2, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--gpus', default=0,
                            help='gpus used for training - e.g 0,1,3')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status') 
        return parser
    
    def forward(self, x):
        x = self.kan(x)
        x = self.rule(x)
        return x
    

    
    def save_weights(self):
        try:
            neural_terms = self.predicate_weights.weight_b.cpu().numpy() # the binary weights do not need gradient 
            print(neural_terms)
            print(self.predicate_weights.weight.detach().numpy())
        except:
            pass
        try:
            neural_predicates = self.rule.interpretable_rul_weights.cpu().numpy() # the rule weights need gradient
            print(neural_predicates)
        except:
            pass
        try:
            if os.path.exists('rule_learning_original/data/weights') == False:
                os.makedirs('rule_learning_original/data/weights')
            np.save('rule_learning_original/data/weights/neural_terms.npy', neural_terms)
            np.save('rule_learning_original/data/weights/neural_predicates.npy', neural_predicates)
        except: 
            pass
        return 0
    
    def interpret(self):
        # atoms = self.obtain_binary_atoms()
        # self.get_symbolic_rules(atoms)
        return 0 
    
    def obtain_binary_atoms(self):
        atoms = collections.defaultdict(list)
        neural_atoms = np.load('rule_learning_original/data/weights/neural_predicates.npy')        
        neural_terms = np.load('rule_learning_original/data/weights/neural_terms.npy')
        threshold = 0
        for threshold in range(0, 100, 5):
            threshold = threshold/100
            activated_index = np.where(neural_atoms > threshold)
            pairs_rule_atoms = [] # store the rule index and highlighted atoms 
            rules_activated_atoms = collections.defaultdict(list)
            for item_index in range(0, len(activated_index[0])):
                pairs_rule_atoms.append((activated_index[0][item_index], activated_index[1][item_index]))
                
            for rule_index,item_index in pairs_rule_atoms:
                rules_activated_atoms[rule_index].append(neural_terms[item_index]) # for each rule, all possible terms, the operator in each rule is conjunction
            
            
            rule = RuleSet()
            for rule_index, all_predicates in rules_activated_atoms.items():
                conjunct = []
                for all_terms in all_predicates: # get the temporal predicate
                    max_weights = np.max(all_terms)
                    max_index = np.where(all_terms == max_weights)
                    min_weights = np.min(all_terms)
                    min_index = np.where(all_terms == min_weights)
                    if min_weights > 0:
                        continue
                    abs_max = np.abs(max_weights)
                    abs_min = np.abs(min_weights)
                    # if abs_max > abs_min:
                        # first_terms = self.feature_name[min_index[0][0]]
                        # second_terms = self.feature_name[max_index[0][0]]
                    # else:
                    first_terms = self.feature_name[max_index[0][0]] # ! how to interpre the predoicate based on the label is made here
                    second_terms = self.feature_name[min_index[0][0]]
                    single_atoms = f'before({first_terms},{second_terms})' # ! how to interpre the predoicate based on the label is made here
                    conjunct.append(single_atoms)
                rule.updates(conjunct)
            atoms[threshold] = rule
        return atoms
    
    def get_symbolic_rules(self,atoms):

        all_con_obj = []
        for threshold, rule_my_obj in atoms.items():
            for conjunctions_my_obj in rule_my_obj.disjunctions:
                predicates = []
                for single_atom in conjunctions_my_obj:
                    predicates.append(Predicate(single_atom, Relation.EQ, 1))
                conj_stand_obj = Conjunction(predicates)
                
                if conj_stand_obj not in all_con_obj:
                    all_con_obj.append(conj_stand_obj)

        rule_set = DnfRuleSet(all_con_obj, 1)
        print(rule_set)
        with open('rule_learning_original/res/rules.md', 'w') as f:
            print(rule_set, file=f)
        return rule_set

    def train_with_kan(self, train_loader, test_loader,opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1., stop_grid_update_step=50, batch=-1,small_mag_threshold=1e-16, small_reg_factor=1., metrics=None, sglr_avoid=False, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video'):
        if loss_fn is None:
            loss_fn = loss_fn_eval = lambda x,y: torch.mean((x-y)**2)
        else:
            loss_fn = loss_fn_eval = loss_fn
        
        if opt == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(self.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
        
        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        results['reg'] = []

            
        grid_update_freq = int(stop_grid_update_step / grid_update_num)
        
        def reg(acts_scale):

            def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
                return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

            reg_ = 0.
            for i in range(len(acts_scale)):
                vec = acts_scale[i].reshape(-1, )

                p = vec / torch.sum(vec)
                l1 = torch.sum(nonlinear(vec))
                entropy = - torch.sum(p * torch.log2(p + 1e-4))
                reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

            # regularize coefficient to encourage spline to be zero
            for i in range(len(self.kan.act_fun)):
                coeff_l1 = torch.sum(torch.mean(torch.abs(self.kan.act_fun[i].coef), dim=1))
                coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(self.kan.act_fun[i].coef)), dim=1))
                reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

            return reg_
        
        global train_loss, reg_
        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(data)
            if sglr_avoid == True:
                id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                train_loss = loss_fn(pred[id_], target[id_])
            else:
                train_loss = loss_fn(pred, target)
            reg_ = reg(self.kan.acts_scale)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective
            
        for epoch_index in range(steps):
            global data, target
            for batch_index, (data, target) in enumerate(train_loader):
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()
                
                if batch_index % grid_update_freq == 0 and batch_index < stop_grid_update_step and update_grid:
                    self.kan.update_grid_from_samples(data)
                
                if opt == "LBFGS":
                    optimizer.step(closure)
                
                if opt == 'Adam':
                    # forward computation
                    pred = self.forward(data)
                    if sglr_avoid == True:
                        id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                        train_loss = loss_fn(pred[id_], target[id_])
                    else:
                        train_loss = loss_fn(pred, target)
                    reg_ = reg(self.kan.acts_scale)
                    loss = train_loss + lamb * reg_
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
            for _, (test_data, test_label) in enumerate(test_loader):
                if self.args.cuda:
                    test_data, test_label = test_data.cuda(), test_label.cuda()
                test_loss = loss_fn_eval(self.forward(test_data), test_label)
                    

            print("epoch %i/%i | train loss: %.2e | test loss: %.2e | reg: %.2e " % (epoch_index, steps, torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(), reg_.cpu().detach().numpy()))

        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__].append(metrics[i]().item())

            results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())

            if save_fig and batch_index % save_fig_freq == 0:
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(batch_index), beta=beta)
                plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()
        # print the trained well parameter in rule 
        with open('./figures/parameters.md', 'w') as f:
            for name, param in self.named_parameters():
                if 'rule' in name:
                    print(param, file=f)
                    print(param)
            f.close()
        return results
    
        

    def train_model(self, train_loader):
        self.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate (train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            output = self.forward(data)
            loss = self.criterion(output, target)
            
            l1_reg = torch.tensor(0., requires_grad=True)
            for name, param in self.named_parameters():
                if 'predicate_weights.weight' in name:
                    l1_reg = l1_reg + torch.linalg.norm(param, 2)

            loss = loss + 10e-2 * l1_reg
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return total_loss/len(train_loader.dataset)
        # return loss.item()

    def test_model(self, test_loader):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = self.forward(data)
                test_loss += self.criterion(output, target).item()
                metric = torcheval.metrics.BinaryAccuracy(threshold=0.5)
                output = torch.reshape(output, [-1])
                target = torch.reshape(target, [-1])
                metric.update(output, target)
                acc = metric.compute()
                correct += acc*len(target)
        test_loss /= len(test_loader.dataset)
        acc = 100. * correct / len(test_loader.dataset)
        return test_loss, acc

    def check_weights(self):
        if os.path.exists('rule_learning_original/data/weights/neural_terms.npy') == True:
            neural_terms = np.load('rule_learning_original/data/weights/neural_terms.npy')
            neural_predicates = np.load('rule_learning_original/data/weights/neural_predicates.npy')
            print(neural_terms, neural_predicates)
            return False
        else:
            return True

    def mul_epochs(self, train_loader, test_loader):
        # train the model
        run_flag  = self.check_weights()
        run_flag = True
        if run_flag == True:
            for epoch in range(1,self.args.epochs+1):
                train_loss = self.train_model(train_loader)
                test_loss, test_acc = self.test_model(test_loader)
                writer.add_scalars('loss_demo', {'train': train_loss, 'test': test_loss}, epoch)
                writer.add_scalar('accuracy_demo/test', test_acc, epoch)
                if epoch%40 == 0:
                    self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']*0.1
                print('Train Loss: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, Epoch: {}/{}'.format(train_loss , test_loss, test_acc, epoch, self.args.epochs))
            # save the weights 
            self.save_weights()
        # interpret the weights
        self.interpret()
        
        return 0
    


if __name__=='__main__':
    writer = SummaryWriter('demo')
    model = learn_minus(number_features=3)  
    torch.cuda.manual_seed(1)
    model.to('cuda:0')
    batch_size = 128
    # read data 
    all_data = pd.read_csv('rule_learning_original/code/neuralBack/event_data.csv')
    train_data = all_data.sample(frac=0.8, random_state=200)
    test_data = all_data.drop(train_data.index)
    
    train_target = torch.tensor(train_data['label'].values.astype(np.float32)).reshape(-1,1)
    train_x = torch.tensor(train_data.drop('label', axis = 1).values.astype(np.float32)) 
    train_tensor = torch.utils.data.TensorDataset(train_x, train_target) 
    train_loader = torch.utils.data.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)


    test_target = torch.tensor(test_data['label'].values.astype(np.float32)).reshape(-1,1)
    test_x = torch.tensor(test_data.drop('label', axis = 1).values.astype(np.float32))
    test_tensor = torch.utils.data.TensorDataset(test_x, test_target)
    test_loader = torch.utils.data.DataLoader(dataset = test_tensor, batch_size = batch_size, shuffle = True)
    
    train_model(model, train_loader, test_loader, 2000, lr=0.01, max_early_stop=30)
    print('ini postive and negative rate')
    pos = test_data[test_data['label'] == 1].shape[0]
    neg = test_data[test_data['label'] == 0].shape[0]
    print(pos, neg, neg/(pos+neg))
    # model.train_with_kan(train_loader, test_loader, opt = "Adam", steps=20, lamb=0.01, lamb_entropy=10., lr=0.01)
    # # model.train_with_kan(train_loader, test_loader, opt = "LBFGS", steps=20, lamb=0.01, lamb_entropy=10., lr=1.)
    # model.kan.plot()
    # model.kan.prune()
    # model.kan.plot(mask=True)
    # model = model.kan.prune()
    # model(train_x.to('cuda:0'))
    # model.plot()
