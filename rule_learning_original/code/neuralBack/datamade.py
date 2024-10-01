from datetime import datetime
from more_itertools import random_product
import random
import os
import numpy as np
import pandas as pd 
import itertools as it
from tqdm import tqdm
import sys
import csv
import pickle
import collections
class DataMade():
    def __init__(self, feature_number = 6, name='') -> None:
        self.feature_number = feature_number
        self.instance_number = 1000
        self.time_point_max = 100 
        self.name = name
        
        
    def generate_one_hot(self):
        data = np.random.randint(0, self.feature_number, self.instance_number)
        a = data
        b = np.zeros((a.size, a.max() + 1))
        b[np.arange(a.size), a] = 1
        print(b)
        self.one_hot = b 
        return 0
    
    def generate_time_binary(self):
        ''' 
        We let feature 1 alway before 3 
        '''
        ini_train_data = []
        for item in self.one_hot:
            max_index = np.argmax(item)
            if max_index == 0:
                new = np.random.randint(0, 33, 1)
            elif max_index == 1:
                new = np.random.randint(33, 66, 1)
            else:
                new = np.random.randint(66, 100, 1)
            list_item = item.tolist()
            list_item.append(new[0])
            ini_train_data.append(list_item)

        self.train_data = np.array(ini_train_data)
        print(self.train_data)
        dataframe = pd.DataFrame(self.train_data, columns=['feature1', 'feature2', 'feature3', 'time'])
        dataframe.to_csv(f"rule_learning_original/code/neuralBack/data.csv", index=False, sep=',')
        return 0
    
    def generate_event_data(self):
        self.feature_number = 3
        data = np.random.rand(self.instance_number,self.feature_number)
        data = np.where(data > 0.5, 1, 0)
        time = np.random.rand(self.instance_number, self.feature_number)
        
        print(data)
        print(time)
        
        feature_data  = np.multiply(data, time)
        label = []
        for item in feature_data:
            first_time = item[0]
            third_time = item[2]
            if first_time > third_time and first_time!=0 and third_time!=0: #! This is the label
                label.append(1)
            else:
                label.append(0)
        columns = ['feature1', 'feature2', 'feature3', 'label']
        new_data = pd.DataFrame(np.concatenate((feature_data, np.array(label).reshape(-1,1)), axis=1), columns=columns)
        print(new_data)
        new_data.to_csv("rule_learning_original/code/neuralBack/event_data.csv", index=False, sep=',')
        return 0    
    
    def generate_multiple_feature_data(self):
        data = np.random.rand(self.instance_number,self.feature_number)
        data = np.where(data > 0.5, 1, 0)
        time = np.random.rand(self.instance_number, self.feature_number)
        
        print(data)
        print(time)
        
        feature_data  = np.multiply(data, time)
        all_label = []
        log_bar = tqdm(feature_data, desc='Generating data')
        for item in feature_data:
            first_time = item[0]
            second_item = item[1]
            label = 0
            if first_time < second_item and first_time!=0 and second_item!=0: #! This is the label
                label = 1
            elif item[2] < item[3] and item[2] != 0 and item[3] != 0: # #3 > #4
                label = 1
            else:
                label = 0
            all_label.append(label)
            log_bar.update(1)
        columns = []
        for i in range(self.feature_number):
            columns.append('feature'+str(i))
        columns.append('label')
        new_data = pd.DataFrame(np.concatenate((feature_data, np.array(all_label).reshape(-1,1)), axis=1), columns=columns)
        print(new_data)
        new_data.to_csv("rule_learning_original/code/neuralBack/event_data.csv", index=False, sep=',')
        return 0    
    def predicate_version_data(self):
        '''
        Extend the DFORL to longer version neural networks and design connective rules with real-valued data 
        '''
        all_combinations = list(it.permutations(range(self.feature_number),2))
        def feature_function(x,y):
            if x > y and x != 0 and y != 0:
                return 1
            else:
                return 0
        feature_names = []
        for item in all_combinations:
            feature_names.append('feature'+str(item[0])+'_'+str(item[1]))
        print(feature_names)
        feature_names.append('label')
        predicate_data = pd.DataFrame(columns=feature_names)
        time_information_data = pd.read_csv("rule_learning_original/code/neuralBack/event_data.csv").to_numpy()
        log_bar = tqdm(time_information_data, desc='Generating predicate data')
        for item in time_information_data:
            log_bar.update(1)
            feature_data = []
            for combination in all_combinations:
                first_time = item[combination[0]]
                second_time = item[combination[1]]
                satisfy = feature_function(first_time, second_time)
                feature_data.append(satisfy)
            label = item[-1]
            feature_data.append(label)
            new = pd.DataFrame(np.array(feature_data).reshape(1,-1), columns=feature_names)
            predicate_data = pd.concat([predicate_data,new])


        print(predicate_data)
        predicate_data.to_csv("rule_learning_original/code/neuralBack/predicate_data.csv", index=False, sep=',')
    
    def generate_boolean(self):
        data = np.random.rand(self.instance_number,self.feature_number)
        data = np.where(data > 0.5, 1, 0)
        print(data)
        
        all_label = []
        log_bar = tqdm(total = data.shape[0], desc='Generating data')
        for item in data:
            if item[0] == 1 and item[2] == 1:
                label = 1
            elif item[1] == 1 and item[3] == 1:
                label = 1
            else:
                label = 0
            all_label.append(label)
            log_bar.update(1)
        train_data = np.concatenate((data, np.array(all_label).reshape(-1,1)), axis=1)
        columns = []
        for i in range(self.feature_number):
            columns.append('feature'+str(i))
        columns.append('label')
        new_data = pd.DataFrame(train_data, columns=columns)
        print(new_data)
        new_data.to_csv(f"rule_learning_original/code/neuralBack/data/boolean_data_{self.name}.csv", index=False, sep=',')

class KGDataMade():

    def __init__(self, all_relation_path, variable_depth, original_data_path, t_relation, res_path,limited_length=None) -> None:
        self.all_relation_path = all_relation_path
        self.variable_number = variable_depth + 2
        self.original_data_path = original_data_path
        self.t_relation = t_relation
        self.res_path = res_path
        self.variable_depth  = variable_depth 
        self.number_to_variable = {0:'X',1:'Y',2:'Z',3:'W',4:'V',5:'U',6:'T',7:'S',8:'R',9:'Q',10:'P',11:'O',12:'N',13:'M',14:'L',15:'K',16:'J',17:'I',18:'H',19:'G',20:'F',21:'E',22:'D',23:'C',24:'B',25:'A'}
        self.limit_length = limited_length

    def classifier(self):
        '''
        Get some information about the relation
        '''
        all_rlation = {}
        save_all_relation = {}
        all_predicate = []
        all_object = {}
        relation = {}
        arity_relation = {}
        pro = {}
        with open(self.all_relation_path,'r') as f:
            new_single_line  = f.readline()        
            while new_single_line:
                one_perd = []
                # retrieval the relation name and two objects
                predicate = new_single_line[0:new_single_line.index(')')]
                if '#' in new_single_line:
                    if 'TEST' in new_single_line:
                        new_single_line = f.readline()
                        continue
                    if 'T.#' in new_single_line or 'T#' in new_single_line:
                        probability = 1
                    else:
                        probability = new_single_line[new_single_line.index('.'):new_single_line.index('#')].replace('.','').replace(' ','').replace('#','')
                        probability = '0.'+probability
                        probability = float(probability) 
                else:
                    probability = 1
                single_line = predicate.split('(')
                relation_name = single_line[0]
                the_rest = single_line[1].split(",")
                first_obj = the_rest[0]
                second_obj = the_rest[1]
                one_perd.append(relation_name)
                one_perd.append(first_obj)
                one_perd.append(second_obj)
                one_perd.append(probability)
                all_predicate.append(one_perd)
                if first_obj not in all_object:
                    all_object[first_obj] = set([])
                if second_obj not in all_object:
                    all_object[second_obj] = set([])
                if relation_name not in all_rlation:
                    all_rlation[relation_name] = [set([]),set([])]
                relation[relation_name] = []
                arity_relation[relation_name] = 1 #  initial arity for all relation are 1
                new_single_line = f.readline()
                pro[predicate+')'] = probability
            f.close()
        all_relation_list = list(all_rlation)
        
        
        # Save all predicate name into the file 
        for i in all_relation_list:
            save_all_relation[i] = i
        with open(self.original_data_path+'/all_relation_dic.dt','wb') as f:
            pickle.dump(save_all_relation,f)
            f.close()
        with open(self.original_data_path+'/all_relation_dic.txt','w') as f:
            print(str(save_all_relation),file = f)
            f.close()  
        
        for pred in all_predicate:
            first_string = str(all_relation_list.index(pred[0])) + '-1'
            second_string = str(all_relation_list.index(pred[0])) + '-2'
            all_object[pred[1]].add(first_string)
            all_object[pred[2]].add(second_string)

        for pred in all_predicate:
            one_tuple = []
            one_tuple.append(pred[1])
            one_tuple.append(pred[2])
            one_tuple = tuple(one_tuple) 
            relation[pred[0]].append(one_tuple) # {'relation_name': (),(),...,()}

            # check the arity of all predicate 
            if pred[1] != pred[2] and arity_relation[pred[0]] == 1:
                arity_relation[pred[0]] = 2
        

        # Make variable - object dictionary
        variable_objects = {} # TODO this dictionary stores the classification of each variables in the task 
        for i in range(self.variable_number):
            variable_objects[i] = set([])
        
        target_arity = arity_relation[self.t_relation]
        if target_arity == 2:     
            for pred in all_predicate:
                if pred[0] == self.t_relation:
                    variable_objects[0].add(pred[1])
                    variable_objects[1].add(pred[2])
            # add variable into the rest of the variable 
            for obj in all_object:
                for variable_name in range(2,self.variable_number):
                    variable_objects[variable_name].add(obj)
        elif target_arity == 1: # TODO. The arity of targetr predicate is 1. In order to make both postive and negative predicate, we ask all variables have the same classification 
            for obj in all_object:
                for variable_name in range(self.variable_number):
                    variable_objects[variable_name].add(obj)
                    # variable_objects[1].add(obj)
                    # variable_objects[2].add(obj)
        
        # Save relation
        with open(self.original_data_path+'/relation_entities.dt','wb') as f:
            pickle.dump(relation, f)
            f.close()
        # Save the probability of all relational facts
        
                    
        with open(self.original_data_path+'/pro.dt','wb') as f:
            pickle.dump(pro,f)
            f.close()
        with open(self.original_data_path+'/pro.txt', 'w') as f:
            print(pro,file=f)
            f.close()
        
        return variable_objects, list(relation.keys()), relation, arity_relation, target_arity

    def get_all_object(self): # This fun is made for country dataset 
        
        all_objects = {}
        objects_1 = [] 
        
        with open(self.original_data_path+'countries.csv') as c_csv:
            csv_reader = csv.reader(c_csv, delimiter=',')
            line = 0
            for row in csv_reader:
                if line == 0:
                    line+=1
                    continue
                row = row[0].split(',')[0].lower().replace(' ','_')
                objects_1.append(row)
                line += 1
            c_csv.close()
        print("👉 All countries are:")
        print(objects_1)

        #Assemble all subregions
        objects_2 = []
        with open(self.original_data_path+'subregions.txt') as f_sub_r:
            sub_r = f_sub_r.read()
            sub_r = sub_r.split('\n')
            objects_2 = sub_r
            f_sub_r.close()    
            
        print("👉 All subrgions are:")
        print(objects_2)
        
        
        objects_3 = []
        with open(self.original_data_path+'regions.txt') as f_r:
            r = f_r.read()
            r = r.split('\n')
            objects_3 = r
            f_r.close()    
            
        print("👉 All rgions are:")
        print(objects_3)
        
        all_objects[1] = objects_1
        all_objects[2] = objects_2
        all_objects[3] = objects_3
        return all_objects


    def get_all_valide_predicate(self, all_substitution, relation, all_variable_permination, t_index):
        '''
        The propositionalization method 2 mentioned by the paper:
        - Check whether there is a preidicate's value is 0 for always, and generated the corresponding valid body
            predicate. 
        '''
        flag_predicates = {}
        relation_index = 0
        relation_name = list(relation.keys())
        target_predicate_index = relation_name.index(self.t_relation)
        print("Target predicate is")
        print(target_predicate_index)
        for i in (relation):
            flag_predicates[relation_index] = [0] * len(all_variable_permination[i])
            relation_index += 1
        sub_index = 0
        
        # generate all trainable data in the formate of: x-y: [[S(x),S(y),S(z)]...][[1,1,0,0]^|number of predicates|...]
        log_bar = tqdm(total = len(all_substitution), desc='Generating data')
        for i in all_substitution:  
            current_relation_index = 0
            for relation_name in relation: #! fix at here 
                current_data = relation[relation_name]
                current_predicate_index = 0
                current_permination = all_variable_permination[relation_name]
                for j in current_permination:
                    first_variable = j[0]
                    second_variable = j[1]
                    target_tuple = []
                    target_tuple.append(i[first_variable])
                    target_tuple.append(i[second_variable])
                    target_tuple = tuple(target_tuple)
                    if target_tuple in current_data:
                        flag_predicates[current_relation_index][current_predicate_index] = 1
                    current_predicate_index += 1
                current_relation_index += 1
            # # Scan the test dataset
            # test_data = test_relation[self.t_relation]
            # test_permination = all_variable_permination[self.t_relation]
            # current_predicate_index = 0
            # for j in test_permination:
            #     first_variable = j[0]
            #     second_variable = j[1]
            #     target_tuple = []
            #     target_tuple.append(i[first_variable])
            #     target_tuple.append(i[second_variable])
            #     target_tuple = tuple(target_tuple)
            #     if target_tuple in test_data:
            #         flag_predicates[target_predicate_index][current_predicate_index] = 1                
            #     current_predicate_index += 1
            
            log_bar.update(1)
            # if sub_index % 2000 == 0:
                # print(sub_index,'/',len(all_substitution), flush=True)   
        
        print("All predicate info is")
        print(flag_predicates)
        valid_index = []
        number_of_predicate = 0
        for i in flag_predicates:
            info = flag_predicates[i]
            for j in info:
                if j == 1 and number_of_predicate != t_index:
                    valid_index.append(number_of_predicate)
                number_of_predicate += 1
                
        # Compute whether all boolean variable in the body are zero
        print("Valid predicare index are")
        print(valid_index)
        template = {}
        number_of_predicate = 0
        for i in flag_predicates:
            template[i] = []
            for j in range(len(flag_predicates[i])):
                if flag_predicates[i][j] == 1 and number_of_predicate != t_index:
                    template[i].append(j)
                number_of_predicate += 1 
                
        print("Valid template")
        print(template)
        return valid_index, template
                
    # def get_test_data(test_relation, test_o1, self.original_data_path,self.t_relation):
        
    #     with open(self.original_data_path+'test.nl','r') as f:
    #         single_line = f.readline()
    #         while single_line:
    #             line = single_line.replace('(',',').replace(')',',').replace('.',',')
    #             line = line.split(',')
    #             fir_e = line[1]
    #             sec_e = line[2]
    #             # if line[0] == line[1]:
    #             #    continue
    #             tem_list = []
    #             tem_list.append(fir_e)
    #             tem_list.append(sec_e)
    #             test_relation[self.t_relation].append(tuple(tem_list))
    #             test_o1.append(fir_e)
    #             single_line = f.readline()
    #         f.close()
            
    #     print("👉  All relation in the test dataset")
    #     print(test_relation)
    #     print(test_o1)
        
                
    def main(self):

        relation_name = []
        variable_objects, relation_name, relation , arity_relation, target_arity = self.classifier()
        
        # revise for scc data 
        # long_obj = {}
        # for item, values in variable_objects.items():
        #     if len(long_obj) < len(values):
        #         long_obj = values
        # for item, values in variable_objects.items():
        #     variable_objects[item] = long_obj

        #template = {} # ! Define the template 
        #template[0] = [0,4] # (0,2),(1,0),(1,2),(2,0),(2,1)
        #template[1] = [1] # (0,1),(0,2),(1,0),(1,2),(2,0),(2,1) neighbo(z,y) never show be one after the substitutation 
        # Assemb all the countries 
        variable_class = {}     # cityLocIn(x,y)  x->countries y->region z->subregion, cities, and regions. 
        # Then possible predicate include: locatedIn[c_2,c_2],locatedIn[c_1,c_2], locatedIn[c_1,c_3], locatedIn[c_2,c_3] neighbor[c_1,c_1]
        for i in variable_objects:
            variable_class[i] = list(variable_objects[i])
            
        # ! debug at here
        ALL_objects = []
        for i in variable_class:
            l = variable_class[i]
            for j in l:
                if j not in ALL_objects:
                    ALL_objects.append(j)

            
        entities_dic_name_number = {}  #sue:0 
        
        for i in ALL_objects:
            entities_dic_name_number[i] = ALL_objects.index(i) 
        print('👉 The second dictionary is', entities_dic_name_number)
        

        print("👉 All relation in the dataset")
        print(relation)

        
        # train the neural predicates
        variable_value = variable_class.values()
        if self.limit_length != None:
            all_substitution = list(it.product(*variable_value))
            random.shuffle(all_substitution)
            all_substitution = all_substitution[0:self.limit_length]
            # all_substitution = np.array(random_product(*variable_value, repeat=self.limit_length))
            # all_substitution = np.reshape(all_substitution, (self.limit_length,-1))
        else:
            all_substitution = list(it.product(*variable_value))
        # all_variable_permination = {}      
        # for i in relation_name:
        #    all_variable_permination[i] = list(itertools.permutations(range(variable_number),2))
        
        # The possible predicate correspodnnig each predicates 
        relation_variable_permutations = {}
        for i in arity_relation:
            if arity_relation[i] == 1:
                empty_list = []
                for variable_index in range(self.variable_number):
                    empty_list.append((variable_index, variable_index))
                relation_variable_permutations[i] = empty_list
            else:
                relation_variable_permutations[i] = list(it.permutations(range(self.variable_number), 2 ))

        # all predicate to list 
        all_atoms = {}
        all_atom_index = 0
        for i in relation_variable_permutations:
            for terms in relation_variable_permutations[i]:
                all_atoms[all_atom_index] = (i, (terms))
                all_atom_index += 1
        print(all_atoms)
        print('Input the selected atoms and split with ,')
        atoms_selected = input()
        if 'all' in atoms_selected:
            pass
        else:
            atoms_selected = atoms_selected.replace(' ','')
            atoms_selected = atoms_selected.split(',')
            atoms_selected = [int(i) for i in atoms_selected]
            new_relation_variable_permutations = collections.defaultdict(list)
            for item in atoms_selected:
                relations = all_atoms[item][0]
                tuples = all_atoms[item][1]
                new_relation_variable_permutations[relations].append(tuples)
            relation_variable_permutations = new_relation_variable_permutations
            
        with open(self.original_data_path+'template.md','a') as f:
            time_str = datetime.today().strftime(f'%Y-%m-%d %H:%M:%S')
            print(time_str, file=f)
            print('all atoms are',all_atoms, file=f)
            print('selected atoms are',atoms_selected, file=f)
            print(relation_variable_permutations, file=f)
            f.close()
        # The index of the target predicate 
        find_index = 0
        for i in range(len(relation_name)):
            for j in relation_variable_permutations[relation_name[i]]:
                if arity_relation[self.t_relation] == 1 and j == (0,0) and i == relation_name.index(self.t_relation):
                    t_index = find_index
                    target_predicate = self.t_relation+"(X,X)"
                elif arity_relation[self.t_relation] == 2 and j == (0,1) and i == relation_name.index(self.t_relation):
                    t_index = find_index
                    target_predicate = self.t_relation+"(X,Y)"
                find_index += 1
                    
        # t_index = relation_name.index(self.t_relation) *  len(list(itertools.permutations(range(variable_number), 2))) + 0
        print('The index of target predicate are:')
        print(t_index)

        
        print('👉 all substitution')
        print(all_substitution[0:5])
        print('👉 all variables')
        print(relation_variable_permutations)
        if os.path.exists(self.res_path+'data/'+self.t_relation+'/')==False:
            os.mkdir(self.res_path+'data/'+self.t_relation+'/')
        with open(self.res_path+'data/' + self.t_relation+'/relation_variable.dt','wb') as f:
            pickle.dump(relation_variable_permutations, f)
            f.close()
        with open(self.res_path+'data/' + self.t_relation+'/relation_variable.txt','w') as f:
            print(relation_variable_permutations, file=f)
            f.close()    

        #begin at here
        train_label = {}
        
        for i in range(len(relation)):
            train_label[i] = []
        
        # Used to store the Boolean label for the taget relational predicate
        target_predicate_index = len(relation)  
        train_label[target_predicate_index] = []
            
        print("👉 Pre-operation on the label dataset:", train_label)
        
        # Save all variable permutations into the corresponding file
        # a_p_a_v = []

        # try:
        #     with open(self.res_path + 'data/'+self.t_relation+'/valid_index.dt', "rb") as f:
        #         res = pickle.load(f)
        #         valid_index = res['valid_index']
        #         template = res['template']
        #         print(res)
        #         f.close()
        # except FileNotFoundError:
        #     valid_index, template= self.get_all_valide_predicate(all_substitution=all_substitution,relation = relation, all_variable_permination=relation_variable_permutations, t_index = t_index)
        #     res = {}
        #     res['valid_index'] = valid_index
        #     res['template'] = template
        #     with open(self.res_path + 'data/'+self.t_relation+'/valid_index.dt', 'wb') as f:
        #         pickle.dump(res, f)
        #         f.close()
        #     with open(self.res_path + 'data/'+self.t_relation+'/valid_index.txt','w') as f:
        #         print(str(res), file = f)
        #         f.close()
        #     print("Save template succeess")

    # ----------------------------------------------------------------
        # open the probabilistic table 
        with open(self.res_path+'data/pro.dt',"rb") as f:
            pro = pickle.load(f)
            f.close()
        
        log_bar = tqdm(total = len(all_substitution), desc='Generating data')
        times_substitution = 0 
        
        all_atoms = []
        for i in relation_variable_permutations:
            for terms in relation_variable_permutations[i]:
                term1 = self.number_to_variable[terms[0]]
                term2 = self.number_to_variable[terms[1]]
                all_atoms.append(i+'['+term1+','+term2+']')
        all_atoms[t_index] = 'label'
                
        for single_substitution in all_substitution:  # generate all trainable data in the formate of: x-y: [[S(x),S(y),S(z)]...][[1,1,0,0]^|number of predicates|...]
            '''
            ALL substitutions based on the all objects 
            '''
            relation_index = 0
            wrong_flag = True  #? The first constrain mentioned in the paper: if all of the input are 0, then the labels should not be one 
            
            # all_body_boolean_value = {} # used to record all boolean values of the body of the logic program

            single_row = []
            for relation_name in relation: #
                # y_one_data = []
                data = relation[relation_name] # Read all relational data 
                current_permination = relation_variable_permutations[relation_name]
                for variable_pair in current_permination:
                    string_tuple = []
                    # begin to save variable arrgement 
                    # if times_substitution == 0:
                        # a_p_a_v.append(variable_pair)
                    for m in variable_pair:
                        string_tuple.append(single_substitution[m])
                    string_tuple = tuple(string_tuple) # like (sue, dinana)...
                    if string_tuple in data:
                        # build the current symbolic relational data 
                        current_fact = relation_name+'('+string_tuple[0] +',' + string_tuple[1]+')'
                        prob_value = pro[current_fact]
                        # y_one_data.append(prob_value)
                        single_row.append(prob_value)
                    else:
                        # y_one_data.append(0)
                        single_row.append(0)
                # all_body_boolean_value[relation_index]=y_one_data
                relation_index += 1   


            body_row = single_row[0:t_index] + single_row[t_index+1:]
            if 1 in body_row:
                wrong_flag = False
            
            # Compute whether all boolean variable in the body are zero
            # check_ind = 0
            # for i in all_body_boolean_value:
            #     label_list=[]
            #     for acq in template[check_ind]:     #Check the tempalte data
            #         label_list.append( all_body_boolean_value[i][acq])
            #     if 1 in label_list:
            #         # ! change to false to check whether accuracy of neural predicate are improved
            #         wrong_flag = False 
            #         break
            #     check_ind += 1
            # target predicate is ancester(x,y) -> ancester(0,1) in the embedding point of view 
            target_tuple = []
            # append the subtitution value corresponding to the variable x
            target_tuple.append(single_substitution[0]) 
            # append the subtitution value corresponding to the variable y
            target_tuple.append(single_substitution[1]) 
            # The symbolic format data correspinding to the variables x and y
            target_tuple = tuple(target_tuple) 

            if target_tuple in relation[self.t_relation] and wrong_flag == True:
                continue
            if times_substitution == 0:
                all_data_csv = [single_row]
            else:
                all_data_csv.extend([single_row])
            
            # # TODO STep 1: Add all non-target predicate into train dataset  
            # for i in all_body_boolean_value:
            #     train_label[i].append(all_body_boolean_value[i])
            
            # # TODO Step 2: Add the target predicare firsr-order feature into the train data
            # if target_tuple in relation[self.t_relation]:
            #     current_fact = self.t_relation+'('+target_tuple[0] +',' + target_tuple[1]+')'
            #     current_fact_pro = pro[current_fact]
            #     train_label[relation_index].append([current_fact_pro])
            # else:
            #     train_label[relation_index].append([0])
            # # if times_substitution % 5000 == 0:
            #     # print(times_substitution,'/',len(all_substitution))
            log_bar.update(1)
            times_substitution += 1

        # train_label[len(relation)+1] = train_label[len(relation)]
        #test_label[len(relation)+1] = test_label[len(relation)]
        
        print("👉 The generated data is:")
        all_data_csv = np.array(all_data_csv)
        all_data_csv = all_data_csv.reshape(-1, len(all_atoms))
        all_data_csv = pd.DataFrame(all_data_csv, columns=all_atoms)
        remove_cols = []
        for col in all_data_csv.columns:
            if all_data_csv[col].max() == all_data_csv[col].min():
                remove_cols.append(col)
        all_data_csv = all_data_csv.drop(columns=remove_cols)
        print(all_data_csv)
        all_data_csv.to_csv(self.res_path + 'data/'+self.t_relation+f'/data{self.variable_depth}.csv', index=False, sep=',')
        self.all_data_csv = all_data_csv
        # print(len(train_label), len(train_label[0]), len(train_label[1]), len(train_label[2]) )
        #print(len(test_x), len(test_label), len(test_label[0]), len(test_label[1]), len(test_label[2]))


        

        # print("👉 Begin to save all variable permutations in all predicates...")
        # with open(self.original_data_path + self.t_relation+'/predicate.txt', 'w') as vf:
        #     vf.write(str(a_p_a_v))
        #     vf.close()
        # with open(self.original_data_path + self.t_relation+'/predicate.list', 'wb') as vf:
        #     pickle.dump((a_p_a_v), vf)
        #     vf.close()
        # print("👉 Save success")

        


        # ? Before the following code, the data in the x is [[1,2,3],[] (all combination of the variable)]
        # ? And the label is {0:[[ the value of first relation under current per],1:[the value for the #2 relation], 3: [the values of target predicticates]]
        # ? After the fillowing code, the data in the label is [[values of #1 relation ],[values of #2]], the label is [[the value of target predicates]]

        
        # final_x = []
        # for i in range(len(train_label) - 1):
        #     final_x.append(train_label[i])

        # new_train_label = train_label[len(train_label) - 1]
        

        # x_file = self.original_data_path + self.t_relation+'/x_train.dt' 
        # y_file = self.original_data_path + self.t_relation+'/y_train.dt' 

        
        # with open(x_file, 'wb') as xf:
        #     pickle.dump(final_x,xf)
        #     xf.close()
        # with open(y_file, 'wb') as yf:
        #     pickle.dump(new_train_label,yf)
        #     yf.close()
        # # with open(x_test_file, 'wb') as xf:
        # #     pickle.dump(final_test_x,xf)
        # #     xf.close()
        # # with open(y_test_file, 'wb') as yf:
        # #     pickle.dump(test_label,yf)
        # #     yf.close()
        
        # all_obj_file = self.original_data_path + self.t_relation+'/all_ent.dt'
        # with open(all_obj_file, 'wb') as f:
        #     pickle.dump(ALL_objects,f)
        #     f.close()    
            
        # # return all the objects in the task 
        # return ALL_objects , target_arity, target_predicate
        return all_data_csv


        


    def gen(self):
        _, target_arity, head_pre = self.main()
        print("-------Generating Data Success!-------")
        return target_arity, head_pre

    def banance_data(self):
        '''
        This function is used to balance the data in the dataset 
        '''
        all_data_csv = pd.read_csv(self.res_path + 'data/'+self.t_relation+f'/data{self.variable_depth}.csv')
        positive_data = all_data_csv[all_data_csv['label'] == 1]
        negative_data = all_data_csv[all_data_csv['label'] == 0]
        print("👉 The number of positive data is:",len(positive_data))
        print("👉 The number of negative data is:",len(negative_data))
        sample_negative = negative_data.sample(n=len(positive_data))
        all_data = pd.concat([positive_data, sample_negative])
        all_data.to_csv(self.res_path + 'data/'+self.t_relation+f'/data{self.variable_depth}_b.csv', index=False, sep=',')
        
        
    



if __name__=="__main__":
    
    # generating propositional rules for testing 
    # data = DataMade(feature_number=100000, name = '100000_features')
    # # print(data.generate_one_hot())
    # # data.generate_time_binary()
    # # data.train_data 
    # # data.generate_multiple_feature_data()
    # # data.predicate_version_data()
    # data.generate_boolean()
    
    # generating KG data for testing 
    
    # args = sys.argv
    # dataset = args[1] 
    # target_relation_name = args[2]
    dataset = 'buzz'
    target_relation_name = 'buzz'
    print(dataset, target_relation_name)
    original_data_path = 'rule_learning_original/code/DFORL/'+dataset+'/data/'
    res_path = 'rule_learning_original/code/DFORL/'+dataset+'/'

    kg = KGDataMade(all_relation_path = original_data_path + f'{target_relation_name}.nl', variable_depth = 1, original_data_path = original_data_path, t_relation = target_relation_name, res_path = res_path, limited_length=None)
    # generate data 
    ALL_objects = kg.main()
    print("👉 The number of all objects in the task:") # 276
    print(len(ALL_objects))
    # sample data 
    # kg.banance_data()