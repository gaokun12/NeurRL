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

## set the argument


class RuleLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    
    def __init__(self, in_size, rule_number, default_alpha = 10):
        super().__init__()
        self.size_in = in_size
        self.rule_number = rule_number
        self.alpha = default_alpha
        weights = torch.Tensor(self.rule_number, self.size_in)
        self.weights = nn.Parameter(weights, requires_grad=True)  # nn.Parameter is a Tensor that's a module parameter.
        nn.init.normal_(self.weights, mean=0.5, std=0.1) # weight init


    def sigmoid_like(self, x):
        y = torch.divide(1,(1+torch.exp(-1 * self.alpha *(x))))  
        return y 
    
    def fuzzy_or(self, x):   
        neg_inputs = 1 - x
        pro_neg_input = torch.prod(neg_inputs, 1)
        predict_value = 1 - pro_neg_input
        predict_value = torch.reshape(predict_value, [-1, 1])
        return predict_value
    
    def forward(self, x):
        self.interpretable_rul_weights = torch.nn.functional.softmax(self.weights)
        clipped_weights = torch.clamp(self.weights, 0, 1)
        w_times_x= torch.mm(x, self.interpretable_rul_weights.t())
        biased = w_times_x - 0.6
        activated_values = self.sigmoid_like(biased)
        rule_satisfy = self.fuzzy_or(activated_values)
        return rule_satisfy # let the rule satisfy be 1 always 
    

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        # if input.size(1) != 784:
        #     input_b=binarized(input)
        self.weight_b=binarized(self.weight)
        out = nn.functional.linear(input,self.weight_b)
        # if not self.bias is None:
        #     self.bias.org=self.bias.data.clone()
        #     out += self.bias.view(1, -1).expand_as(out)

        return out


    
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
    def __init__(self, number_features=3, number_predicate=2):
        super(NeuralPredicate, self).__init__()
        parser = self.argument()
        self.args = parser.parse_args()
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.set_device(0)
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.args.cuda else {}
        
        self.number_features =  number_features
        self.number_predicates = number_predicate
        self.predicate_weights = BinarizeLinear(self.number_features, self.number_predicates, bias=False)
        self.relu1 = nn.ReLU()
        self.rule = RuleLayer(self.number_predicates, 4)
        self.feature_name = {}
        for i in range(self.number_features):
            self.feature_name[i] = f'feature_{i}'
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam([
                {'params': self.rule.parameters()},
                {'params': self.predicate_weights.parameters(), 'lr': 0.0001}
            ], lr=0.0001, weight_decay=0.9)
    
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
        parser.add_argument('--no-cuda', action='store_true', default=True,
                            help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=2, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--gpus', default=0,
                            help='gpus used for training - e.g 0,1,3')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status') 
        return parser
    
    def forward(self, x):
        x = self.predicate_weights(x)
        x = self.relu1(x)
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
    
class Binary_interpret(NeuralPredicate):
    def __init__(self, number_features=3, number_predicate=2):
        super(Binary_interpret, self).__init__(number_features, number_predicate)
        
    def forward(self,x):
        x = self.predicate_weights(x)
        x = self.relu1(x)
        return x
    

if __name__=='__main__':
    writer = SummaryWriter('demo')
    model = NeuralPredicate(number_features=3, number_predicate=2)  
    args = model.args
    
    # read data 
    all_data = pd.read_csv('rule_learning_original/code/neuralBack/event_data.csv')
    train_data = all_data.sample(frac=0.8, random_state=200)
    test_data = all_data.drop(train_data.index)
    
    train_target = torch.tensor(train_data['label'].values.astype(np.float32)).reshape(-1,1)
    train_x = torch.tensor(train_data.drop('label', axis = 1).values.astype(np.float32)) 
    train_tensor = torch.utils.data.TensorDataset(train_x, train_target) 
    train_loader = torch.utils.data.DataLoader(dataset = train_tensor, batch_size = args.batch_size, shuffle = True)


    test_target = torch.tensor(test_data['label'].values.astype(np.float32)).reshape(-1,1)
    test_x = torch.tensor(test_data.drop('label', axis = 1).values.astype(np.float32))
    test_tensor = torch.utils.data.TensorDataset(test_x, test_target)
    test_loader = torch.utils.data.DataLoader(dataset = test_tensor, batch_size = args.batch_size, shuffle = True)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # model.mul_epochs(train_loader, test_loader)
    
    model = Binary_interpret(number_features=3, number_predicate=1)
    model.mul_epochs(train_loader, test_loader)

