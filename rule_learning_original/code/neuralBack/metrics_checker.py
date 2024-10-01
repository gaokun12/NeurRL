from pyDatalog import pyDatalog 
import pickle

class CheckMetrics():
    def __init__(self,t_relation, task_name, classifier, logic_path,t_arity=2, test_mode=False, cap_flag = False, ruleset = None):
        self.t_relation = t_relation 
        self.task_name = task_name
        self.test_mode = test_mode
        self.cap_flag = cap_flag
        self.data_path = 'rule_learning_original/code/DFORL/'+self.task_name+'/data/'
        self.result_path = 'rule_learning_original/code/DFORL/' + self.task_name + '/data/' + self.t_relation +'/'
        self.logic_path = logic_path
        self.hit_flag = False
        self.t_arity=t_arity
        self.hit_test_predicate = []
        self.classifier = classifier
        self.rule_set = ruleset
        if self.cap_flag == False:
            self.facts_path = self.data_path+self.t_relation+'.nl'
        else:
            self.facts_path = self.data_path+self.t_relation+'.nl'
        self.all_variables = ['X','Y','Z','W','M','N','T','U','V','A','B','C','D','E','F','G','H','I','J','K','L','O','P','Q','R','S']
        self.all_variables_str = 'X,Y,Z,W,M,N,T,U,V,A,B,C,D,E,F,G,H,I,J,K,L,O,P,Q,R,S,'
        if self.t_arity == 2:
            self.first_variable = 'X'
            self.second_variable = 'Y'
        elif self.t_arity == 1:
            self.first_variable = 'X'
            self.second_variable = 'X'
        else:
            print('The arity of the target relation is not supported')
        
    def  build_target_predicate(self):
        '''
        build target predicate and return a dictionary consisting the predicate 
        '''
        all_target_predicate_dic = {}
        with open(self.facts_path, 'r') as f:
            single_line = f.readline()
            while single_line:
                # skip the negative examples in the datasets, we won't add the negative examples in test datasets
                if '-' in single_line:
                    single_line = f.readline()
                    continue
                if self.test_mode == True:
                    # if the test mode is True, then check the accuracy based on the test datasets 
                    if 'TEST' in single_line:
                        head_relation = single_line[:single_line.index('(')]
                        if head_relation == self.t_relation:
                            all_target_predicate_dic[single_line[:single_line.index(')')+1]] = 0
                else:
                    # if the test mode is Off, then check auucracy expecting the test datasets
                    if 'TEST' in single_line:
                        single_line = f.readline()
                        continue
                    head_relation = single_line[:single_line.index('(')]
                    if head_relation == self.t_relation:
                        all_target_predicate_dic[single_line[:single_line.index(')')+1]] = 0
                single_line = f.readline()
            f.close()
        self.all_target_predicate_dic = all_target_predicate_dic
        return all_target_predicate_dic
    
    def check_recall_of_logic_program(self, test_mode = False):
        '''
        Input the symbolic logic program, return the correctness of each rule in the logic prgoram.
        '''
        # expresison_list, variable_index, head_relation = build_datalog_base_and_expression(data_path, t_relation, logic_program_name, task_name, result_path)
        
        relation_path = self.data_path + 'all_relation_dic.dt'
        with open(relation_path,'rb') as f:
            all_relation = pickle.load(f)
            f.close()
        # if classifer is None, then there is no rule
        try:
            rule_list = self.classifier.rules
        except:
            rule_list = []
        head_relation = self.t_relation
        precision_rule = [] 
        body = []
        for rule in rule_list:
            if self.hit_flag == True:
                precision_rule.append(rule.precision)
            body.append(str(rule.conjunction))
                
        # for rule in rule_list:
        #     body = []
        #     while rule:
        #         if '#' in rule:
        #             latter_part = rule[rule.index('#'):]
        #             rule = rule[:rule.index('#')]
        #             probability = float(latter_part[latter_part.index('(')+1:latter_part.index(',')])
        #             precision_rule.append(probability)
        #         one_body = []
        #         rule = rule.replace(' ','')
        #         rule = rule.replace('-','')
        #         rule = rule.split(':')
                
        #         body_relation = rule[1]
                
        #         single_body = body_relation.split('&')
        #         for item in single_body:
        #             one_body.append(item)

                
        #         body.append(one_body)
        #         rule = f.readline()[:-2]
        #     f.close()

        # ! build the variable into Datalog
        term_log = ''
        # term_log += 'X,Y,Z,W,M,N,T,'
        term_log += self.all_variables_str
        for i in all_relation:
            term_log += i + ','
            
        term_log = term_log[0:-1]
        pyDatalog.create_terms(term_log)
        
        with open(self.facts_path, 'r') as f:
            single_tri = f.readline()
            while single_tri:
                # skip the negative examples in the datasets
                # if  '@' in single_tri and '-' not in single_tri:
                if '-' in single_tri: # Do not build the negative examples
                    single_tri = f.readline()
                    continue
                # Do not add any test data in the file
                # !During the training process, do not inclue the testing facts. 
                # !But during the testing process, containing the testing facts.
                if test_mode == False and 'TEST' in single_tri:
                    single_tri = f.readline()
                    continue
                #prepare objects
                single_tri = single_tri[:single_tri.index('.')]
                relation_name = single_tri[:single_tri.index('(')]
                first_entity = single_tri[single_tri.index('(') + 1 : single_tri.index(',')]
                second_entity = single_tri[single_tri.index(',')+1 : single_tri.index(')')]
                #add to database
                + locals()[relation_name](first_entity,second_entity)
                
                single_tri = f.readline()
            f.close()
        
        # Check each generated rules 
        expresison_list = []
        variable_index = []
        for rules in body:
            # Find the order of variables x and y and z in the rules
            va_f = 0
            o_variable_index = []
            for var in self.all_variables:
            # for var in ['X','Y','Z','W','M','N','T']:
                if var in rules:
                    index = rules.index(var)
                    va_f = 1
                else:
                    index = 1e6
                o_variable_index.append((var, index))
            
            o_variable_index.sort(key= lambda y:y[1])
            
            var_dic = {}
            for i in range(len(o_variable_index)):
                var_dic[o_variable_index[i][0]] = i
            if va_f == 1:
                variable_index.append(var_dic)
            flag = 0
            item_list = rules.replace(' ','').split('^')
            for item in item_list:
                item = item[1:-1]
                negation_flag = False
                if item == '':
                    continue
                else:
                    name = item[:item.index('[')]
                    # if the neagation operator in the rule 
                    if '~' in name:
                        negation_flag = True
                        name = name[1:]
                    first_variable = item[item.index('[')+1: item.index(',')].upper()
                    second_variable = item[item.index(',')+1: item.index(']')].upper()
                    if flag == 0:
                        if negation_flag == True:
                            expression =  ~(locals()[name](locals()[first_variable],locals()[second_variable]))
                        else:
                            expression =  locals()[name](locals()[first_variable],locals()[second_variable])
                    else:
                        if negation_flag == True:
                            expression &=  ~(locals()[name](locals()[first_variable],locals()[second_variable]))
                        else:
                            expression &=  locals()[name](locals()[first_variable],locals()[second_variable])
                    flag += 1
            if flag != 0:
                expresison_list.append(expression)
        # ! each expression corresponds a rule
        # read the target predicate file 
        if self.hit_flag == True:
            target_dic = self.hit_test_predicate
        else:
            target_dic = self.build_target_predicate()
        
        # The following check process follows the T_P operator of the logic program 
        correct_f = []
        search_index = 0 
        for res in expresison_list: # expression_list: [[[g(x),g(y),g(z)],[g(x),g(y),g(z)],[g(x),g(y),g(z)]]...]
            num_validate = 0
            correct = 0
            for re in res:
                x_index = variable_index[search_index][self.first_variable]
                y_index = variable_index[search_index][self.second_variable]
                if x_index >= len(re) or y_index >= len(re):
                    break
                num_validate += 1
                if self.t_arity == 2:
                    first_res = re[x_index]
                    # if first_res == 'iraq':
                    #     b = 90
                    second_res = re[y_index]
                elif self.t_arity == 1:
                    first_res = re[x_index]
                    second_res = re[x_index]
                final = len(locals()[head_relation](first_res,second_res)) # The ground predicate 
                # ! test mode open iff when check accuracy basedon the .nl file
                if test_mode == True:
                    predicate = head_relation + '(' + first_res+',' +second_res+ ')'
                    if predicate in target_dic:
                        if self.hit_flag == True:
                            current_precision_value = target_dic[predicate]
                            # iff the new precision are larger than the current one, updata the value 
                            if precision_rule[search_index] >= current_precision_value: 
                                target_dic[predicate] = precision_rule[search_index]
                        else:
                            target_dic[predicate] = 1
                        correct += 1    
                # ! when the test mode is false and the target ground predicate in the database
                elif final == 1:  
                    predicate = head_relation + '(' + first_res+',' +second_res+ ')'
                    if predicate in target_dic:
                        target_dic[predicate] = 1
                    correct += 1
            if num_validate == 0:
                num_validate = -1
            correct_f.append((correct/num_validate ,correct, num_validate))
            search_index += 1
        print(correct_f)
        # write the state of target predicate into the disk 
        # When executing single task, writing each test predicate  logic in the disk 
        # if self.hit_flag == False:
        #     write_target_predicate(task_name, target_dic, t_relation)
        false = 0
        for key in target_dic:
            if target_dic[key] == 0:
                false += 1
        recall_KG = (len(target_dic) - false)/len(target_dic)
        if correct_f != [] and recall_KG != 0:
            with open(self.logic_path, 'a') as f:
                print('**precision** from KG:', correct_f, file= f)
                print(f'**recall** from KG:**{recall_KG}**', file=f)
                f.close()
            
        # if hit == True:
        #     with open(data_path +  t_relation + '/relation_entities.dt', 'rb') as f:
        #         relation_entity = pickle.load(f)
        #         f.close()
        #     res = calculate_Hits(target_tuple,expresison_list, correct_f, variable_index)
        #     print("Hits result:", res)
        
        if self.hit_flag == True:
            correct_f = target_dic
        return_obj  = {'precision':correct_f, 'recall':recall_KG}
        print('from KG', return_obj)
        return return_obj

    def check_logic_program_with_IP(self, test_mode = False):
        '''
        check the inventied logic program with the IP operator
        '''
        # expresison_list, variable_index, head_relation = build_datalog_base_and_expression(data_path, t_relation, logic_program_name, task_name, result_path)
        
        relation_path = self.data_path + 'all_relation_dic.dt'
        with open(relation_path,'rb') as f:
            all_relation = pickle.load(f)
            f.close()
        # when check from predicated invention need to do that 
        rule_list = self.rule_set
        head_relation = self.t_relation
        body = ''
        for rule in rule_list:
            all_conjuntions = rule
            for conj in all_conjuntions:
                body += (str(conj))+'^'
        body = [body[:-1]]
        # ! build the variable into Datalog
        term_log = ''
        term_log += self.all_variables_str
        # add existing predicates
        for i in all_relation:
            term_log += i + ','

        
        term_log = term_log[0:-1]
        pyDatalog.create_terms(term_log)
        
        with open(self.facts_path, 'r') as f:
            single_tri = f.readline()
            while single_tri:
                # skip the negative examples in the datasets
                # if  '@' in single_tri and '-' not in single_tri:
                if '-' in single_tri: # Do not build the negative examples
                    single_tri = f.readline()
                    continue
                # Do not add any test data in the file
                # !During the training process, do not inclue the testing facts. 
                # !But during the testing process, containing the testing facts.
                if test_mode == False and 'TEST' in single_tri:
                    single_tri = f.readline()
                    continue
                #prepare objects
                single_tri = single_tri[:single_tri.index('.')]
                relation_name = single_tri[:single_tri.index('(')]
                first_entity = single_tri[single_tri.index('(') + 1 : single_tri.index(',')]
                second_entity = single_tri[single_tri.index(',')+1 : single_tri.index(')')]
                #add to database
                + locals()[relation_name](first_entity,second_entity)
                
                single_tri = f.readline()
            f.close()
        
        # Check each generated rules 
        expresison_list = []
        variable_index = []
        for rules in body:
            # Find the order of variables x and y and z in the rules
            va_f = 0
            o_variable_index = []
            for var in self.all_variables:
                if var in rules:
                    index = rules.index(var)
                    va_f = 1
                else:
                    index = 1e6
                o_variable_index.append((var, index))
            
            o_variable_index.sort(key= lambda y:y[1])
            
            var_dic = {}
            for i in range(len(o_variable_index)):
                var_dic[o_variable_index[i][0]] = i
            if va_f == 1:
                variable_index.append(var_dic)
            flag = 0
            item_list = rules.replace(' ','').split('^')
            for item in item_list:
                negation_flag = False
                if item == '':
                    continue
                else:
                    name = item[:item.index('[')]
                    # if the neagation operator in the rule 
                    if '~' in name:
                        negation_flag = True
                        name = name[1:]
                    first_variable = item[item.index('[')+1: item.index(',')].upper()
                    second_variable = item[item.index(',')+1: item.index(']')].upper()
                    if flag == 0:
                        if negation_flag == True:
                            expression =  ~(locals()[name](locals()[first_variable],locals()[second_variable]))
                        else:
                            expression =  locals()[name](locals()[first_variable],locals()[second_variable])
                    else:
                        if negation_flag == True:
                            expression &=  ~(locals()[name](locals()[first_variable],locals()[second_variable]))
                        else:
                            expression &=  locals()[name](locals()[first_variable],locals()[second_variable])
                    flag += 1
            if flag != 0:
                expresison_list.append(expression)
        
        # ! each expression corresponds a rule
        # read the target predicate file 
        if self.hit_flag == True:
            target_dic = self.hit_test_predicate
        else:
            target_dic = self.build_target_predicate()
        
        # derived atoms from then invented predicate 
        derived_atoms = []
        for res in expresison_list:
            for re in res:
                x_index = variable_index[0]['X']
                y_index = variable_index[0]['Y']
                if x_index >= len(re) or y_index >= len(re):
                    break
                if self.t_arity == 2:
                    first_res = re[x_index]
                    second_res = re[y_index]
                elif self.t_arity == 1:
                    first_res = re[x_index]
                    second_res = re[x_index]
                derived_atoms.append(self.t_relation+f'({first_res},{second_res})')
        if set(derived_atoms).issubset(set(target_dic.keys())):
            print('invented predicate success')
            return 1
        else:
            print('invented predicate not success')
            return 0
        # The following check process follows the T_P operator of the logic program 
        # correct_f = []
        # search_index = 0 
        # for res in expresison_list: # expression_list: [[[g(x),g(y),g(z)],[g(x),g(y),g(z)],[g(x),g(y),g(z)]]...]
        #     num_validate = 0
        #     correct = 0
        #     for re in res:
        #         x_index = variable_index[search_index]['X']
        #         y_index = variable_index[search_index]['Y']
        #         if x_index >= len(re) or y_index >= len(re):
        #             break
        #         num_validate += 1
        #         if self.t_arity == 2:
        #             first_res = re[x_index]
        #             # if first_res == 'iraq':
        #             #     b = 90
        #             second_res = re[y_index]
        #         elif self.t_arity == 1:
        #             first_res = re[x_index]
        #             second_res = re[x_index]
        #         final = len(locals()[head_relation](first_res,second_res)) # The ground predicate 
        #         # ! test mode open iff when check accuracy basedon the .nl file
        #         if test_mode == True:
        #             predicate = head_relation + '(' + first_res+',' +second_res+ ')'
        #             if predicate in target_dic:
        #                 target_dic[predicate] = 1
        #                 correct += 1    
        #         # ! when the test mode is false and the target ground predicate in the database
        #         elif final == 1:  
        #             predicate = head_relation + '(' + first_res+',' +second_res+ ')'
        #             if predicate in target_dic:
        #                 target_dic[predicate] = 1
        #             correct += 1
        #     if num_validate == 0:
        #         num_validate = -1
        #     correct_f.append((correct/num_validate ,correct, num_validate))
        #     search_index += 1
        # print(correct_f)
        # # write the state of target predicate into the disk 
        # # When executing single task, writing each test predicate  logic in the disk 
        # # if self.hit_flag == False:
        # #     write_target_predicate(task_name, target_dic, t_relation)
        # false = 0
        # for key in target_dic:
        #     if target_dic[key] == 0:
        #         false += 1
        # recall_KG = (len(target_dic) - false)/len(target_dic)
        # with open(self.logic_path, 'a') as f:
        #     print('**precision** from KG:', correct_f, file= f)
        #     print(f'**recall** from KG:**{recall_KG}**', file=f)
        #     f.close()
            
        # # if hit == True:
        # #     with open(data_path +  t_relation + '/relation_entities.dt', 'rb') as f:
        # #         relation_entity = pickle.load(f)
        # #         f.close()
        # #     res = calculate_Hits(target_tuple,expresison_list, correct_f, variable_index)
        # #     print("Hits result:", res)
        
        # if self.hit_flag == True:
        #     correct_f = target_dic
        # return correct_f
