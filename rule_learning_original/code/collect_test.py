# import pickle
# import time
# import itertools
import numpy as np 
import pandas as pd
# from copy import deepcopy
# import tensorflow as tf 
import os 
# from tslearn.utils import to_time_series_dataset
# from tslearn.clustering import TimeSeriesKMeans
# from dtaidistance import dtw
# import pandas as pd 
# from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import sys
sys.path.append('/home/gaokun/kk/')
sys.path.append('/home/gaokun/kk/aix360_kun/aix360/')
sys.path.append('/home/gaokun/kk/aix360_kun/')
# from aix360_kun.aix360.algorithms.rule_induction.ripper import RipperExplainer
import aix360_kun.aix360.algorithms.rule_induction.trxf.classifier.ruleset_classifier as trxf_classifier
# import matplotlib.pyplot as plt
# import matplotlib 
# matplotlib.use('Agg')
# from collections import defaultdict, namedtuple
# from colorama import Fore
# import tqdm
# from shutil import copyfile
# import math 
from datetime import datetime 
from aix360_kun.aix360.algorithms.rule_induction.trxf.core.conjunction import Conjunction
from aix360_kun.aix360.algorithms.rule_induction.trxf.core.predicate import Predicate, Relation
from aix360_kun.aix360.algorithms.rule_induction.trxf.core.dnf_ruleset import DnfRuleSet
from aix360_kun.aix360.algorithms.rule_induction.trxf.core.feature import Feature
from aix360_kun.aix360.algorithms.rule_induction.trxf.core.utils import batch_evaluate
# each times series can corresponds a training instance

class Collect():
    def __init__(self, task_name='gun', rule_learner = 'ripper', father_path = 'rule_learning_original', test_path = 'sgh_10point_mulevent_limited_train_l5_s1_c10',rule_path = None, result_name = None) -> None:
        self.task_name = task_name
        self.father_path = father_path
        self.rule_learner = rule_learner
        if rule_path == None:
            self.rule_path = os.path.join(self.father_path, 'res', f'{self.task_name}', 'dforl.md')
        else:
            self.rule_path = rule_path+'/dforl.md'
        self.data_path = os.path.join(self.father_path, 'data', f'{test_path}/event_test.csv')
        self.default_label = 1.0
        if result_name == None:
            self.result_name = os.path.join(self.father_path, 'res', f'{self.task_name}')
        else: 
            self.result_name = result_name
    
    def read_rule(self, remove_low_precision = None):
        # read rules from rule_path
        with open(self.rule_path, 'r') as f:
            all_rules = f.read()
            f.close()
        all_conjunctions = set([])
        all_rules = all_rules.split('\n')
        for line in all_rules:
            if '==' in line:
                line = line.replace(' ', '')
                atoms = line.split('precision')[0]
                precision  = float(line.split('precision')[1].split('recall')[0][1:])
                if remove_low_precision != None:
                    if precision <remove_low_precision:
                        continue
                all_conjunctions.add(atoms)
        self.conjunction_string = all_conjunctions
        
    def make_rule_obj(self):
        # make rule object
        all_conjunctions = []
        for clause in self.conjunction_string:
            new_clause = clause.replace(' ', '')
            new_clause = new_clause.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
            atoms = new_clause.split('^')
            all_predicate = []
            for atom in atoms:
                predicate = atom.split('==')[0]
                value = atom.split('==')[1]
                if value == '1':
                    print(predicate, value)
                    all_predicate.append(Predicate(feature=Feature(predicate), relation=Relation.EQ, value=1))
            conjunction = Conjunction(all_predicate)
            all_conjunctions.append(conjunction)
        self.rule = DnfRuleSet(all_conjunctions, then_part=True)
        return self.rule

    
    def update_metrics(self):
        test_data = pd.read_csv(self.data_path)
        X = test_data.drop('label', axis=1)
        y = test_data['label']        
        classifier = trxf_classifier.RuleSetClassifier([self.rule],rule_selection_method=trxf_classifier.RuleSelectionMethod.WEIGHTED_MAX,confidence_metric=trxf_classifier.ConfidenceMetric.LAPLACE,weight_metric=trxf_classifier.WeightMetric.CONFIDENCE,default_label=self.default_label)
        classifier.update_rules_with_metrics(X, y)
        y_pred = batch_evaluate(self.rule, X)
        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        accuracy_rule_set = (tn+tp)/(tn+fp+fn+tp)
        precision_rule_set = tp/(tp+fp)
        recall_rule_set = tp/(tp+fn)
        tnr = tn/(tn+fp)
        f1 = 2*precision_rule_set*recall_rule_set/(precision_rule_set+recall_rule_set)
        
        
        
        all_rules = str(self.rule)
        rules_str = all_rules.replace('v','')
        rules_str = rules_str.split('\n')[1:-2]
        new_rule = []
        precision = []
        recall = []
        for rule in classifier.rules:
            precision.append(rule.precision)
            recall.append(rule.recall)
            
        for index,item in enumerate(rules_str):
            new_rule.append(item + f'precision: {precision[index]} recall: {recall[index]}' + '\n')
        
        with open(self.result_name+f'/original.md', 'a+') as f:
            print(f'**{datetime.now()}**', file=f)
            print(f'Accuracy: {accuracy_rule_set}', file=f)
            print(f'Precision: {precision_rule_set}', file=f)
            print(f'Recall: {recall_rule_set}', file=f)
            print(f'TNR: {tnr}', file=f)
            print(f'F1: {f1}', file=f)
            print(f'Confusion Matrix: tn {tn}, fp {fp}, fn {fn}, tp {tp}', file=f)
            for i in new_rule:
                print(i, file=f)
            f.close()
        
        return 1
    
    
    
if __name__ == '__main__':
    collect = Collect(task_name='sgh_4point_mulevent_1819_abn_DFORL', rule_learner='', father_path='rule_learning_original',test_path='sgh_4point_mulevent_1819_abn_l5_s1_c20')
    collect.read_rule(remove_low_precision=0.8)
    collect.make_rule_obj()
    collect.update_metrics()
    
    