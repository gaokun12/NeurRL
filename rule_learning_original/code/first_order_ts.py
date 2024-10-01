import itertools
import os
import pickle
import sys
import time
from copy import deepcopy
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from dtaidistance import dtw
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score)
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
sys.path.append(os.getcwd())
from rule_learning_original.code.collect_test import Collect
import matplotlib
import matplotlib.pyplot as plt

import aix360_k.aix360.algorithms.rule_induction.trxf.classifier.ruleset_classifier as trxf_classifier
from aix360_k.aix360.algorithms.rule_induction.ripper import RipperExplainer
from aix360_k.aix360.algorithms.rule_induction.trxf.core.dnf_ruleset import \
    DnfRuleSet
from aix360_k.aix360.algorithms.rule_induction.trxf.metrics import \
    compute_rule_metrics

matplotlib.use('Agg')
import math
from collections import defaultdict, namedtuple
from datetime import datetime
from shutil import copyfile

import tqdm
from colorama import Fore
# each times series can corresponds a training instance
from parameter import return_parameter
from neuralBack.dforl_torch import RuleLayer
import neuralBack.main as dforl
from aix360_k.aix360.algorithms.rule_induction.trxf.core.conjunction import \
    Conjunction
from rule_learning_original.code.dkm.main import *


class Feature():
    ''''
    The class for clustering data and event data
    '''
    def __init__(self, data, start_time, end_time, clustering_index, time_zones = -1) -> None:
        self.data = data
        self.start_time = start_time
        self.end_time = end_time
        self.c_index = clustering_index
        self.time_zones = time_zones
    
    def calculate_length(self):
        length = self.end_time - self.start_time
        return length

class SingleRuleHighlights():
    def __init__(self, highdata, series_index ,precision, recall, rule_str) -> None:
        self.single_high_slices = highdata
        self.valid_timeseries_index = series_index
        # rule information appears here
        self.precision = precision
        self.recall = recall
        self.rule_str = rule_str
        self.event_data = []
    def add_event_data(self, event_data):
        self.event_data.extend(event_data)
        return 0

single_highlights_slice = namedtuple('single_highlights_slice', ['data', 'start_index', 'color'])

single_highlights_slice_with_end = namedtuple('single_highlights_slice_with_end', ['data', 'start_index', 'end_index', 'color'])

class RuleFormat():
    def __init__(self) -> None:
        # initialize each continue clustering attribute and event features into black list first. 
        self.series = []# (clustring variable 1, time zone 1)
        self.event  = []# (event variable 1, time zone 1)
        self.series_data_index = [] # store the corresponding data which can be described by this rule 
        self.rule_single_obj = None # the rule object
        self.recall = 0
        self.precision = 0
        self.lift = 0
        self.recall_subgroup = 0 # the recall in the subgroup
        self.precision_subgroup = 0 
        self.tp = 0 
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.tp_with_X = 0
        self.fp_with_X = 0
        self.tn_with_X = 0
        self.fn_with_X = 0
        
    def update_rule(self, rule:dict):
        for key, value in rule.items():
            if  key == 'series':
                self.series.append(value)
            elif key == 'E':
                self.event.append(value)
        return 0
    
    def update_p_r(self, p, r, l=0):
        self.precision = p
        self.recall = r
        self.lift = l
        return 0

    def update_series_data_index(self, index):
        self.series_data_index.append(index)
    
    def compute_recall_precision_on_subgroups(self, test_data, label_name):
        '''
        Compute the recall and precision for subgroups 
        '''
        # select subgroup data by index 
        subgroups = test_data.iloc[self.series_data_index]
        x_test = subgroups.drop(columns=[label_name])
        y_test = subgroups[label_name]
        # compute the recall and precision for subgroups
        try:
            metric_list = compute_rule_metrics(self.rule_single_obj, x_test, y_test)
        except:
            self.precision_subgroup = 'no instances meet body'
            self.recall_subgroup = 'no instances meet body'
            self.tp_with_X = 'no instances meet body'
            self.fp_with_X = 'no instances meet body'
            self.tn_with_X = 'no instances meet body'
            self.fn_with_X = 'no instances meet body'
            return 0
        self.precision_subgroup = metric_list[0].precision
        self.recall_subgroup = metric_list[0].recall
        self.tp_with_X = metric_list[0].tp
        self.fp_with_X = metric_list[0].fp
        self.tn_with_X = metric_list[0].tn
        self.fn_with_X = metric_list[0].fn
        
        

class tsDWTmeans():
    def __init__(self, data_map_train, n_clustering, secrio_path) -> None:
        self.data_map_train = data_map_train
        self.n_clustering = n_clustering
        self.secrio_path = secrio_path

    def fit(self):
        if os.path.exists(f'rule_learning_original/data/{self.secrio_path}/tsdwt.p'):
            self.clustering_model = pickle.load(open(f'rule_learning_original/data/{self.secrio_path}/tsdwt.p', 'rb'))
            return self.clustering_model
        all_subsquences = []
        for i in self.data_map_train:
            single_time_series_part = i[1]
            for seq in single_time_series_part:
                seq = np.array(seq, dtype=float)
                seq = seq[~np.isnan(seq)]
                seq = list(seq)
                if len(seq) <= 1:
                    continue
                all_subsquences.append(seq)
        print(len(all_subsquences))
        train_list = to_time_series_dataset(all_subsquences)
        start_time = time.time()
        self.clustering_model = TimeSeriesKMeans(n_clusters=self.n_clustering, metric="dtw",verbose=True).fit(train_list)
        ent_time = time.time()
        print('Training DWT Clustering Time (sec):'+str(ent_time - start_time))
        if os.path.exists(f'rule_learning_original/data/{self.secrio_path}') == False:
            os.mkdir(f'rule_learning_original/data/{self.secrio_path}')
        pickle.dump(self.clustering_model, open(f'rule_learning_original/data/{self.secrio_path}/tsdwt.p', 'wb'))

    
    def predict(self, all_data):
        return self.clustering_model.predict(all_data)
    
    
class DWT_model():
    def __init__(self, data_map_train, secrio_path) -> None:
        self.data_map_train = data_map_train
        self.secrio_path = secrio_path
        self.threshold_dwt = 10 
        
    def fit(self):
        all_subsquences = {}
        item = 0
        for i in self.data_map_train:
            single_time_series_part = i[1]
            for seq in single_time_series_part:
                seq = np.array(seq, dtype=float)
                non_nan_index = ~np.isnan(seq)
                seq_no_nan = seq[non_nan_index]
                if len(seq_no_nan)==0:
                    continue
                all_subsquences[(str(item),)] = [seq_no_nan]
                item += 1
        print(len(all_subsquences))
        start_time = time.time()
        clustering_model = self.clustering(all_subsquences)
        ent_time = time.time()
        print('Training DWT Clustering Time (sec):'+str(ent_time - start_time))
        ini_index_clustering = 0
        for item, data in clustering_model.items():
            clustering_model[item] = (data, ini_index_clustering)
            ini_index_clustering += 1
        self.clustering_model = clustering_model
        return clustering_model
    
    
    def predict(self, data_test):
        '''
        data_test is a list of data
        '''
        clustering_index = []
        allo_log = tqdm.tqdm(total=len(data_test), desc='Allocating Clustering')
        for item in data_test:
            single_index = self.dwt_clustering_finding(item, self.clustering_model)[0]
            clustering_index.append(single_index)
            allo_log.update(1)
        return clustering_index
    
    def dwt_clustering_finding(self,single_data, trajectoriesSet):
        '''
        For single data to find the clustering index 
        '''
        non_nan_index = ~np.isnan(single_data)
        single_data = single_data[non_nan_index]
        
        min_clustering = -1
        min_metric = 1000000
        
        for filter, trajectory in trajectoriesSet.items():
            for subItem in trajectory[0]:
                single_min_distance = 1000000
                metric = dtw.distance(single_data, subItem, psi=1)
                if metric < single_min_distance: 
                    single_min_distance = metric
            if single_min_distance <= min_metric:
                min_clustering = trajectory[1] # clustering index 
                min_metric = single_min_distance
        return min_clustering, min_metric
    
    def clustering(self, trajectoriesSet):
        if os.path.exists(f'rule_learning_original/data/{self.secrio_path}/trajectories.p'):
            trajectories = pickle.load(open(f'rule_learning_original/data/{self.secrio_path}/trajectories.p', 'rb'))
            return trajectories
        trajectories = deepcopy(trajectoriesSet)
        distanceMatrixDictionary = {}
        iteration = 0
        ini_length = len(trajectories)
        ini_log = tqdm.tqdm(total=ini_length, desc='DWT Clustering Ini', position=0)
        while True:            
            iteration += 1
            distanceMatrix = np.empty((len(trajectories), len(trajectories),))
            distanceMatrix[:] = np.nan
            if iteration == 1:
                single_distance_computing_log = tqdm.tqdm(total=len(trajectories), desc='First DWT Clustering Computing')
            for index1, (filter1, trajectory1) in enumerate(trajectories.items()):
                tempArray = []
                for index2, (filter2, trajectory2) in enumerate(trajectories.items()):
                    if index1 > index2:
                        continue
                    elif index1 == index2:
                        continue
                    else:
                        unionFilter = filter1 + filter2
                        sorted(unionFilter)
                        if unionFilter in distanceMatrixDictionary.keys():
                            distanceMatrix[index1][index2] = distanceMatrixDictionary.get(unionFilter)
                            continue
                        metric = []
                        for subItem1 in trajectory1:
                            for subItem2 in trajectory2:
                                metric.append(dtw.distance(subItem1, subItem2, psi=1))
                        metric = max(metric)
                        distanceMatrix[index1][index2] = metric
                        distanceMatrixDictionary[unionFilter] = metric  
                if iteration == 1:
                    single_distance_computing_log.update(1)
            minValue = np.min(list(distanceMatrixDictionary.values()))
            maxValue = np.max(list(distanceMatrixDictionary.values()))
            if minValue > self.threshold_dwt:
                break
            minIndices = np.where(distanceMatrix == minValue)
            minIndices = list(zip(minIndices[0], minIndices[1]))
            minIndex = minIndices[0]
            filter1 = list(trajectories.keys())[minIndex[0]]
            filter2 = list(trajectories.keys())[minIndex[1]]
            trajectory1 = trajectories.get(filter1)
            trajectory2 = trajectories.get(filter2)
            unionFilter = filter1 + filter2
            sorted(unionFilter)
            trajectoryGroup = trajectory1 + trajectory2
            trajectories = {key: value for key, value in trajectories.items()
                            if all(value not in unionFilter for value in key)}
            distanceMatrixDictionary = {key: value for key, value in distanceMatrixDictionary.items()
                                        if all(value not in unionFilter for value in key)}
            trajectories[unionFilter] = trajectoryGroup
            ini_log.set_description(f'min:{minValue} max {maxValue} thre {self.threshold_dwt}  continue? {minValue < self.threshold_dwt}')
            ini_log.update(1)
        pickle_file = f'rule_learning_original/data/{self.secrio_path}/trajectories.p'
        pickle.dump(trajectories, open(pickle_file, 'wb'))
        
        
        # for item, data in trajectories.items():
        #     for single in data:
        #         plt.plot(single)
        #     plt.show()
        # print('one')
        return 1 
    
class TimeSeries:
    def __init__(self, rule_learning_method='ripper',train_data_path = '', test_data_path = '', result_name='', task_name='', father_path = '', unit= 1, low=4, high=11, clustering_method = 'knn', max_plot=50, min_precision=0.8, min_recall=0.8, max_inter = 10, multi_class_label = None, n_clustering = 5) -> None:

        self.subsequence_length = 5 # for clustering 
        self.stride = 1 # for clustering 
        self.n_clustering = n_clustering #! for sgh = 20, for ucr = 5, need to updated, coffee = 10

        # get clustering names  
        self.clustering_name = ['series'+str(i) for i in range(self.n_clustering)]

        
        self.single_zone_length = 0
        
        self.data_split_senario = f'{task_name}_l{self.subsequence_length}_s{self.stride}_c{self.n_clustering}'
        self.father_path = father_path
        
        
        self.record_unit_time = unit # with hours as the unit
        # set the label name and default value 
        self.task_name=task_name
        self.setting_parameters(task_name)
        
        self.result_name = result_name
        if os.path.exists(self.result_name) == False:
            os.makedirs(self.result_name)
        self.rule_learning_method = rule_learning_method
        
        self.train_path = train_data_path
        self.test_path = test_data_path
        
        self.plot_max = max_plot
        self.color = ['-ro','-go','-bo','-yo','-co','-mo','-ko', '-wo', '-ro','-go','-bo','-yo','-co','-mo','-ko', '-wo', '-ro','-go','-bo','-yo','-co','-mo','-ko', '-wo', '-ro','-go','-bo','-yo','-co','-mo','-ko', '-wo', '-ro','-go','-bo','-yo','-co','-mo','-ko', '-wo', '-ro','-go','-bo','-yo','-co','-mo','-ko', '-wo', '-ro','-go','-bo','-yo','-co','-mo','-ko', '-wo', '-ro','-go','-bo','-yo','-co','-mo','-ko', '-wo', '-ro','-go','-bo','-yo','-co','-mo','-ko', '-wo', ]
        self.color_fore = [Fore.RED, Fore.GREEN, Fore.BLUE, Fore.YELLOW, Fore.CYAN, Fore.MAGENTA, Fore.BLACK, Fore.WHITE]
        self.temporal_relation = {'before': lambda x,y: x < y}
        self.low = low
        self.high = high
        
        self.clustering_method = clustering_method
        
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.max_inter_rule = max_inter
        
        self.multi_class_label = multi_class_label
        

        
    def setting_parameters(self, task_name):
        self.default_label, self.default_negative, self.label_name, self.time_event_name, self.time_series_name, self.body_windows_length, self.future_window_length = return_parameter(task_name)
        return 1    
        
    # def move_label_to_first_row(self, data_path=None):
    #     '''
    #     Move the label to the first row
    #     '''
    #     data = pd.read_csv(data_path)
    #     label = data[self.label_name]
    #     data = data.drop(columns=[self.label_name])
    #     data = pd.concat([label, data], axis=1)
    #     data.to_csv(data_path, index=False)
    #     return data    

    def update_data_map(self):
        pass    
    
    def clustering(self):
        if self.clustering_method == 'knn':
            return self.knn() # return a model 
        elif self.clustering_method == 'dwt':
            dwt_model = DWT_model(self.data_map_train, self.data_split_senario)
            dwt_model.fit()
            self.n_clustering = len(dwt_model.clustering_model)
            return dwt_model # return a model
        elif self.clustering_method == 'tsDWTmeans':
            tsDWTmeans_model = tsDWTmeans(self.data_map_train, self.n_clustering, self.data_split_senario)
            tsDWTmeans_model.fit()
            return tsDWTmeans_model
    

    def data_process(self, move_label=True):
        if move_label == True:
            self.row_train = self.move_label_to_first_row(self.train_path)
            self.row_test = self.move_label_to_first_row(self.test_path)
        
        self.data_map_train = self.split_subsequence(self.row_train, length = self.subsequence_length, stride = self.stride)
        self.data_map_test = self.split_subsequence(self.row_test, length = self.subsequence_length, stride = self.stride)
        
        # get k means
        self.kmeans = self.clustering()
        
        self.update_data_map()

        # get the clustering feature data
        self.x_train,self.label_train = self.obtain_rules_format(self.data_map_train,'train')
        self.x_test,self.label_test = self.obtain_rules_format(self.data_map_test,'test')
        
        # combining the short data into longer data
        self.x_aggregate_train = self.aggregate_data(self.x_train)
        self.x_aggregate_test = self.aggregate_data(self.x_test)
        # ! key condition 
        if os.path.exists(os.path.join(self.father_path, 'data', self.data_split_senario, 'train.csv')) == False:
            # 
            print('Begin to preprocess the data')
            # build the temporal relations
            temporal_pd_train = self.self_temporal_features(self.x_aggregate_train, 'train')
            temporal_pd_test= self.self_temporal_features(self.x_aggregate_test, 'test')
            
            # build the clustering feature trainable data
            train_pd = self.build_clustering_occurrence(self.x_aggregate_train, self.label_train)
            test_pd = self.build_clustering_occurrence(self.x_aggregate_test, self.label_test)
            
            # combining the clustering feature data and temporal relationship data
            self.train_pd = self.combine_train_pd(train_pd, temporal_pd_train)
            self.test_pd = self.combine_train_pd(test_pd, temporal_pd_test)
            

            de_log = tqdm.tqdm(total=len(self.train_pd.columns), desc='Delete Columns with only one value')
            delete_columns = []
            for col in self.train_pd.columns:
                if len(self.train_pd[col].unique()) == 1:
                    delete_columns.append(col)
                de_log.update(1)
            self.train_pd = self.train_pd.drop(columns=delete_columns)            
            self.test_pd = self.test_pd.drop(columns=delete_columns)
            print(f'Delete {len(delete_columns)} Columns on training and test data')

            
            self.train_pd.to_csv(os.path.join(self.father_path, 'data', self.data_split_senario, 'train.csv'), index=False)
            self.test_pd.to_csv(os.path.join(self.father_path, 'data', self.data_split_senario, 'test.csv'), index=False)
        else:
            self.train_pd = pd.read_csv(os.path.join(self.father_path, 'data', self.data_split_senario,'train.csv'))
            self.test_pd = pd.read_csv(os.path.join(self.father_path, 'data', self.data_split_senario,'test.csv'))
            
    def only_plot_rules(self):
        self.data_process()
        collect = Collect(task_name=self.task_name, rule_learner='', father_path='rule_learning_original',test_path=self.data_split_senario, rule_path=self.result_name, result_name=self.result_name)
        collect.read_rule(remove_low_precision=0.8)
        self.rules = collect.make_rule_obj()
        collect.update_metrics()
        p, r, l = self.compute_precision_recall(self.rules)
        try:
            self.format_rule_atom = self.transfer_rule_to_atom(self.rules, p, r, l) # the rule in our formats with sub precision, recall, series data, and event data
        except:
            print('No rules are found')
            return 0
        # analyzing whether there are temporal relationship between founded clustering features
        # self.generate_temporal_relations(format_rule_atom)
        rule_highlight = self.obtain_explainable_figure(self.format_rule_atom, self.x_test, self.data_map_test)
        self.plot(rule_highlight, self.test_path)
        self.plot_some_negative(self.test_path)
        self.get_sub_precision_recall()
        return 0 

    def run(self):
        self.data_process()
        self.rules = self.rule_method() # the rule object 
        p, r, l = self.compute_precision_recall(self.rules)
        try:
            self.format_rule_atom = self.transfer_rule_to_atom(self.rules, p, r, l) # the rule in our formats with sub precision, recall, series data, and event data
        except:
            print('No rules are found')
            return 0
        # analyzing whether there are temporal relationship between founded clustering features
        # self.generate_temporal_relations(format_rule_atom)
        rule_highlight = self.obtain_explainable_figure(self.format_rule_atom, self.x_test, self.data_map_test)
        self.plot(rule_highlight, self.test_path)
        self.plot_some_negative(self.test_path)
        self.get_sub_precision_recall()
        return 0 
    
    def get_sub_precision_recall(self):
        with open(self.result_name+f'/original.md', 'a+') as f:
            print('------------------',file=f)
            for single_format_rule in self.format_rule_atom:
                single_format_rule.compute_recall_precision_on_subgroups(self.test_pd, self.label_name)
                print(str(single_format_rule.rule_single_obj.conjunctions[0]), file=f)
                print(f'with X->Y, subgroups instances with X: precision: {single_format_rule.precision_subgroup}  recall: {single_format_rule.recall_subgroup} *tp: {single_format_rule.tp_with_X} fp: {single_format_rule.fp_with_X} tn: {single_format_rule.tn_with_X} fn: {single_format_rule.fn_with_X}*', file=f)
                print(f'with X->Y, all       instances with X: precision: {single_format_rule.precision} recall: {single_format_rule.recall} lift: {single_format_rule.lift}', file=f)
                print('\n', file=f)
            f.close()
        return 0

    def aggregate_data(self, data_set):
        '''
        For single variable, only glucose is recorded in this stage 
        '''
        continues_clustering_data = []
        for data in data_set:
            single_data = []
            start_index = 0
            for item_index, item in enumerate(data):
                if item == -1:
                    start_index = item_index + 1
                    # start_index = self.stride * item_index + 1 # for the stride is not 1 
                    continue
                if self.single_zone_length != 0: # TODO same clustering should extend naturally, and the time also should be extended naturally. 
                    time_zone = int(start_index/self.single_zone_length)
                else:
                    time_zone = -1
                current_clustering = Feature([],start_index, item_index+1, 'series'+str(item), time_zones='at'+str(time_zone))
                try:
                    next_item = data[item_index+1]
                    if next_item == item:
                        continue
                    else:
                        single_data.append(current_clustering)
                        start_index = item_index + 1
                        # start_index = self.stride * item_index + 1 # for the stride is not 1 
                except:
                    single_data.append(current_clustering)
            continues_clustering_data.append(single_data)
        return continues_clustering_data
    
    
    def self_temporal_features(self, data, log_info):
        all_pairs_clustering = itertools.product(self.clustering_name, self.clustering_name)
        all_temporal_atoms = {}
        for predicate in self.temporal_relation:
            for item in all_pairs_clustering:
                all_temporal_atoms[predicate+'_'+str(item[0])+'_'+str(item[1])+'_'] = 0
        trainable_data_temporal = pd.DataFrame(columns=all_temporal_atoms.keys())
        all_atoms = set(all_temporal_atoms.keys())
        temporal_relation_bar = tqdm.tqdm(total=len(data),desc=f'Temporal Relation on {log_info}',ascii=True)
        for item in data:
            all_temporal_atoms = dict.fromkeys(all_temporal_atoms, 0)
            all_pairs_clustering = itertools.permutations(item, 2)
            all_list = list(all_pairs_clustering)
            for c_pairs in all_list:
                first_index = c_pairs[0].c_index
                second_index = c_pairs[1].c_index
                first_end = c_pairs[0].end_time-1
                second_start = c_pairs[1].start_time
                for predicate, predicate_function in self.temporal_relation.items():
                    if predicate_function(first_end, second_start):
                        atom = predicate+'_'+str(first_index)+'_'+str(second_index)+'_'
                        if atom in all_atoms:
                            all_temporal_atoms[atom] = 1
                        else:
                            raise ValueError('The atom is not in the keys')
                    else:
                        atom = predicate+'_'+str(first_index)+'_'+str(second_index)+'_'
                        if atom in all_atoms:
                            all_temporal_atoms[atom] = 0
                        else:
                            raise ValueError('The atom is not in the keys')
            trainable_data_temporal = trainable_data_temporal._append(all_temporal_atoms, ignore_index=True)
            temporal_relation_bar.update(1)
        return trainable_data_temporal
                    
            
            
    def combine_train_pd(self, *args, **kwargs):
        final_pd = pd.concat(args, axis=1)
        return final_pd
                    
        
    def obtain_long_feature(self, single_format_rule, data_feature, data_original_with_map):
        '''
        This function is used to find the temporal relations between existing clustering features.
        Inout:
        data_feature: self.x_train or self.x_tet
        data_original_with_map: self.data_map_train or self.data_map_test
        Output:
        return the temporal relationships between different features, which is pandas format, trainable by RIPPER to find first-order rules with these temporal relationships 
        '''
        label = []
        x_for_each_rule = []
        all_permutations = itertools.permutations(single_format_rule[0], 2)
        all_column = []
        all_atoms = {}
        for predicate in self.temporal_relation:
            for items in all_permutations:
                single_atom = predicate + '(' + str(items[0]) + ',' + str(items[1]) + ')'
                target_tuple = (predicate, items[0], items[1])
                all_column.append(single_atom)
                all_atoms[target_tuple] = 0
        all_column.append('label')
        temporal_finding_train_data = pd.DataFrame(columns=all_column)
        
        for time_series_index, time_series_single in enumerate(data_feature):
            single_data = []
            label = data_original_with_map[time_series_index][0]
            for feature in single_format_rule[0]:   
                start_time = 0
                end_time = 0
                index = 0
                while index < len(time_series_single):
                    cluster = time_series_single[index]
                    if cluster == feature:
                        start_time = index 
                        end_time = index + 1
                        while end_time < len(time_series_single) and time_series_single[end_time] == feature:
                            end_time += 1
                        single_clustering = Feature([], start_time, end_time, feature)
                        single_data.append(single_clustering)
                        index = end_time
                        continue
                    index += 1
                            
            # build temporal relationship between different clustering features

            all_permutations = itertools.permutations(single_data, 2)             
            for predicate, predicate_function in self.temporal_relation.items():
                for pairs_permutations in  all_permutations:
                    atom = (predicate,pairs_permutations[0].c_index, pairs_permutations[1].c_index)
                    # exclude the self relationship
                    if atom not in all_atoms.keys():
                        continue
                    first_item_time = pairs_permutations[0].start_time 
                    second_item_time = pairs_permutations[1].start_time
                    if predicate_function(first_item_time, second_item_time):
                        all_atoms[atom] = 1
            for key, value in all_atoms.items():
                key_str = key[0] + '(' + str(key[1]) + ',' + str(key[2]) + ')'
                temporal_finding_train_data.loc[time_series_index,key_str] = value
            temporal_finding_train_data.loc[time_series_index,'label'] = label
        # ! There is  no duplicated clustering features during plot the figure. Need to check why there are duplicated features here. 
        # ! need to consider all temporal relationships between possible subsequence 
        return temporal_finding_train_data
    
    def obtain_explainable_figure(self, format_rule_atom, cluster_data, estimated_data):
        '''
        input: rules, test time series original data, knn data, 
        output: for each rule, generate a folder to store the explainable figure
        '''
        rule_highlight = defaultdict(list)
        str_rules = []
        for rule in self.rules.conjunctions:
            str_rules.append(str(rule))
        for time_series_index, time_series_cluster in enumerate(cluster_data):
            unique_knn_cluster = np.unique(time_series_cluster)
            # only check the positive label time series data
            if estimated_data[time_series_index][0] == 0:
                continue
            for rule_index,rule in enumerate(format_rule_atom):
                # if all positive literals in the rule 
                rule_glucose = rule.glucose
                # TODO if satisfied the rule
                if set(rule_glucose).issubset(set(unique_knn_cluster)):
                    highlight_slices = []
                    for clustering_index, cluster in enumerate(time_series_cluster):
                        if cluster in rule_glucose:
                            atom_index = rule_glucose.index(cluster)
                            single_highlight_slice = single_highlights_slice(estimated_data[time_series_index][1][clustering_index], clustering_index, self.color[atom_index])
                            # highlight.append([estimated_data[time_series_index][1][clustering_index],clustering_index, self.color[atom_index]])
                            highlight_slices.append(single_highlight_slice)
                        else:
                            continue
                    rule_highlight_single = SingleRuleHighlights(highlight_slices, time_series_index, rule.precision, rule.recall, str_rules[rule_index])
                    rule_highlight[str(tuple(rule_glucose))].append(rule_highlight_single)
                    # rule_highlight[str(tuple(rule))].append([highlight, time_series_index, rule_p_r[1], rule_p_r[2], str_rules[rule_index]])
        return rule_highlight
    
    def plot(self, rule_highlight, test_data_path):
        # the first row is column name
        test_data = pd.read_csv(test_data_path) 
        plot_path = self.result_name+'/rule_plot'
        if os.path.exists(plot_path) == False:
            os.makedirs(plot_path)

        rule_index = 0 
        for clustering_in_rule, estimated_value in rule_highlight.items():
            rule_path = os.path.join(plot_path, str(clustering_in_rule))
            if os.path.exists(rule_path) == False:
                os.makedirs(rule_path)
            for data_index, all_plot in enumerate(estimated_value):
                if data_index >= self.plot_max:
                    break
                time_series_index = all_plot.valid_timeseries_index
                time_series_raw = test_data.iloc[time_series_index][1:] # the first one is label
                fig = plt.figure(figsize=(10,6))
                ax = fig.add_subplot(111)
                time_series_raw = np.array(time_series_raw, dtype=float)
                

                total_length = self.body_windows_length + self.future_window_length
            
                plt.plot(range(0, self.body_windows_length, self.record_unit_time), time_series_raw,'-D',label=f'full time series {data_index}')
                for item in all_plot.single_high_slices:
                    start_index = item.start_index
                    sub_data = item.data
                    color_plot = item.color
                    plt.plot(range(start_index, start_index+self.subsequence_length, self.record_unit_time), sub_data, color_plot)
                title_str = ''
                list_key = clustering_in_rule[1:-1].replace(' ','').split(',')
                for index,i in enumerate(list_key):
                    title_str +=  f'{self.color[index]}({i}) '
                ax.set_title(f'Rule: {all_plot.rule_str} \n {title_str}, p: {round(all_plot.precision,2)}, r: {round(all_plot.recall,2)}, \n No. {str(data_index)} time series data')
                    #  , first {self.color[0]}, second {self.color[1]}, third {self.color[2]}, fourth {self.color[3]}',fontsize=12)

                plt.xticks(np.arange(total_length+1),np.arange(self.body_windows_length,-1*self.future_window_length-1,-1), rotation=90)

                if self.low != None and self.high != None:
                    plt.hlines(y=self.low, xmin=0, xmax=total_length, linestyles='--', colors='k')
                    plt.hlines(y=self.high, xmin=0, xmax=total_length, linestyles='--', colors='k')
                plt.savefig(os.path.join(rule_path, str(data_index)+'.png'))
                plt.close()
            rule_index += 1
        return 0 
    
    def plot_some_negative(self, test_data_path):
        # the first row is column name
        test_data = pd.read_csv(test_data_path) 
        plot_path = self.result_name+'/rule_plot'
        if os.path.exists(plot_path) == False:
            os.makedirs(plot_path)
        try:
            negative_data = test_data[test_data[self.label_name] == self.default_negative].sample(n=self.plot_max)
        except:
            negative_data = test_data[test_data[self.label_name] == 0]
        
        negative_data = negative_data.to_numpy()
        
        rule_path = os.path.join(plot_path, 'negative')
        if os.path.exists(rule_path) == False:
            os.makedirs(rule_path)
        total_length = self.body_windows_length + self.future_window_length
        rule_index = 0 
        plot_log = tqdm.tqdm(total=len(negative_data), desc='Plotting Negative Data')
        for data_index, all_plot in  enumerate(negative_data):
            plot_log.update(1)
            if 'sgh' in self.task_name or 'mimic' in self.task_name:
                content = all_plot[1:25]
            else:
                content = all_plot[1:]
            fig = plt.figure(figsize=(20,6))
            ax = fig.add_subplot(111)
            # ! no event plot here 
            plt.plot(range(0, self.body_windows_length, self.record_unit_time), content,'-D',label=f'negative series: {data_index}')
            ax.set_title(f'Negative No. {str(data_index)} time series data')
            plt.xticks(np.arange(total_length+1),np.arange(self.body_windows_length,-1*self.future_window_length-1,-1), rotation=90)

            if self.low != None and self.high != None:
                plt.hlines(y=self.low, xmin=0, xmax=total_length, linestyles='--', colors='k')
                plt.hlines(y=self.high, xmin=0, xmax=total_length, linestyles='--', colors='k')
            plt.savefig(os.path.join(rule_path, str(data_index)+'.png'))
            plt.close()
            rule_index += 1
        return 0 
    
    def transfer_rule_to_atom(self, rules, p, r, l):
        '''
        Transfer the rules to atoms
        '''
        rules = str(rules)
        all_rules = []
        rules_str = rules.replace('v','')
        rules_str = rules_str.split('\n')[1:-2]
        for rule_index, item in enumerate(rules_str):
            single_rule = RuleFormat()
            items_conjunction = item.split('^')
            for i in items_conjunction:
                # only select positive literal in the rule
                if '== 1' in i and 'before' not in i:
                    i = i.replace(' ','').replace('[','').replace(']','').replace('(','').replace(')','')
                    if 'series' in i:
                        start_index = i.find('series') + len('series')
                        # ! need to update the index finding process
                        end_index = i.find('=')
                        sub_string = i[start_index: end_index]
                        atom_index = int(sub_string)
                        single_rule.update_rule({'series':atom_index})
                    else:
                        end_index = i.find('=')
                        sub_string = i[:end_index]
                        single_rule.update_rule({'E':sub_string})

                    # single_rule.append(atom_index)
                elif 'before' in i and '== 1' in i:
                    three_tuple = i.split('_')
                    first_atom = (three_tuple[1])
                    second_atom = (three_tuple[2]) 
                    for atom in [first_atom, second_atom]:
                        if 'series' in atom:
                            start_index = atom.find('series') + len('series')
                            sub_string = atom[start_index:]
                            atom_index = int(sub_string)
                            single_rule.update_rule({'series':atom_index})
                        elif 'E' in atom:
                            single_rule.update_rule({'E':atom})
                    # i = i.replace('series','')
                    # atom_pairs = i.replace(' ', '')
                    # start_index = atom_pairs.find('_')+1
                    # end_index = atom_pairs.find('=')
                    # sub_string = atom_pairs[start_index: end_index]
                    # sub_string = sub_string.split('_')
                    # first_atom = int(sub_string[0])
                    # second_atom = int(sub_string[1])
                    # single_rule.append(first_atom)
                    # single_rule.append(second_atom)
                # elif 'before' in i and '== 1' in i and 'E' in i:
                #     i = i.replace('E','')
                #     atom_pairs = i.replace(' ', '')
                #     start_index = atom_pairs.find('_')+1
                #     end_index = atom_pairs.find('=')
                #     sub_string = atom_pairs[start_index: end_index]
                #     sub_string = sub_string.split('_')
                #     first_atom = int(sub_string[0])
                #     second_atom = int(sub_string[1])
                #     single_rule.append(first_atom)
                #     single_rule.append(second_atom)
            single_rule.update_p_r(p[rule_index], r[rule_index], l[rule_index])
            all_rules.append(single_rule)
        return all_rules
    
    def compute_precision_recall(self, rule_set, sub_rule=''):
        '''
        Based on the testing datasets, computing the accuracy, precision, recall for each rules 
        '''
        
        y_test = self.test_pd[self.label_name]
        x_test = self.test_pd.drop(columns=[self.label_name])
        
        classifier = trxf_classifier.RuleSetClassifier([rule_set],rule_selection_method=trxf_classifier.RuleSelectionMethod.WEIGHTED_MAX,confidence_metric=trxf_classifier.ConfidenceMetric.LAPLACE,weight_metric=trxf_classifier.WeightMetric.CONFIDENCE,default_label=self.default_label)
        classifier.update_rules_with_metrics(x_test, y_test)
        
        precision = []
        recall = []
        tp = []
        tn = []
        fp = []
        fn = []
        lift = []

        for rule in classifier.rules:
            precision.append(rule.precision)
            recall.append(rule.recall)
            tp.append(rule.tp)
            tn.append(rule.tn)
            fp.append(rule.fp)
            fn.append(rule.fn)
            lift.append(rule.lift)
            
        all_rules = str(rule_set)
        rules_str = all_rules.replace('v','')
        rules_str = rules_str.split('\n')[1:-2]
        new_rule = []
        try:
            for index,item in enumerate(rules_str):
                new_rule.append(item + f'precision: {precision[index]} recall: {recall[index]} lift {lift[index]} tp: {tp[index]} tn: {tn[index]} fp: {fp[index]} fn: {fn[index]}' + '\n')
        except:
            print('No rule')
            
        
        with open(self.result_name+f'/original{sub_rule}.md', 'a+') as f:
            for i in new_rule:
                print(i, file=f)
            f.close()
        return precision, recall, lift
    
    def obtain_rules_format(self, data_map, flag):
        '''
        output: [element_number 1, ..., element_number n] [label] for one time series data
        '''
        label = []
        x = []
        number_data = len(data_map)
        each_instance_data = len(data_map[0][1])
        number_point = len(data_map[0][1][0])
        all_x = []
        all_label = []
        for i in data_map:
            single_time_series_part = i[1]
            single_data = []
            for data in single_time_series_part:
                single_data.append(data)
            all_x.append(single_data)
            all_label.append(i[0])
        
        all_x = np.array(all_x)
        all_x = np.reshape(all_x, (number_data*each_instance_data, number_point))
        if os.path.exists(os.path.join('rule_learning_original','data', self.data_split_senario)) == False:
            os.mkdir(f'rule_learning_original/data/{self.data_split_senario}/')
        if os.path.exists(f'rule_learning_original/data/{self.data_split_senario}/clustering_label_{flag}.p'):
            clustering_label = pickle.load(open(f'rule_learning_original/data/{self.data_split_senario}/clustering_label_{flag}.p', 'rb'))
        else:            
            clustering_label = []
            for item in tqdm.tqdm(all_x):
                try:
                    clustering_label.append(self.kmeans.predict([item])[0])
                except:
                    clustering_label.append(-1) # represent the all nan value
            # clustering_label = self.kmeans.predict(all_x)
            pickle.dump(clustering_label, open(f'rule_learning_original/data/{self.data_split_senario}/clustering_label_{flag}.p', 'wb'))
        label = np.reshape(clustering_label, (number_data, each_instance_data))
        original_label = np.array(all_label)
        
        # for i in data_map:
        #     single_time_series_part = i[1]
        #     single_data = []
        #     for data in single_time_series_part:
        #         k_index = self.kmeans.predict([data])
        #         single_data.append(k_index)
        #     label.append(i[0])
        #     single_data = np.array(single_data).reshape(-1)
        #     x.append(single_data)
        # x= np.array(x).reshape(-1,len(single_data))
        return label, original_label
    def change_data_label(self):
        '''
        Change the label to the multi-class label
        '''
        tn = self.task_name.split('_')[1]
        train_label = pd.read_csv(self.multi_class_label+f'{tn}_train.csv')
        test_label = pd.read_csv(self.multi_class_label+f'{tn}_test.csv')
        self.train_pd[self.label_name] = train_label[self.label_name]
        self.test_pd[self.label_name] = test_label[self.label_name]
        self.row_train[self.label_name] = train_label[self.label_name]
        self.row_test[self.label_name] = test_label[self.label_name]
        index= 0
        for item in self.data_map_test:
            item[0] = test_label[self.label_name][index]
            index += 1
        print('change label success')
        return 0

        
    def rule_method(self):
        '''
        Return rules in IBM xmlp format
        '''
        if self.multi_class_label != None:
            self.change_data_label()
            
        if self.rule_learning_method == 'ripper':
            rule, metrics_test = self.ripper_learning(self.train_pd, self.test_pd)
        elif 'DFORL' in self.rule_learning_method:
            rule, metrics_test = self.DFOL_learning(self.train_pd, self.test_pd, self.min_precision, self.min_recall, self.max_inter_rule)
        else:
            raise ValueError('Invalid rule learning method')
        if os.path.exists(self.result_name) == False:
            os.makedirs(self.result_name)
        with open(self.result_name+'/original.md', 'a+') as f:
            f.write('**'+str(datetime.now())+'**\n')
            f.write(f'clustering length {self.subsequence_length}, stride {self.stride}, clustering method {self.clustering_method}, rule learning method {self.rule_learning_method}, past windows {self.body_windows_length}, future windows {self.future_window_length}\n')
            # f.write(str(rule)+'\n\n')
            for key, value in metrics_test.items():
                f.write(key + ' ' + str(value) + '\n')
            f.close()
        return rule

    def split_subsequence(self, data, length = 10, stride = 1):
        # using a map to store the data and value
        '''
        input: file_path
        output: [label, [[s1],[s2]...,[sn]]] for one time series data
        '''
        data_map = []
        for index, row in data.iterrows():
            label = row[0]
            row_data = list(row[1:])
            data_pool = []
            # remove the data which is not enough for the length
            for i in range(0, len(row_data), stride):
                if len(row_data[i:i+length]) < length:
                    break
                data_pool.append(row_data[i:i+length])
            data_map.append([label, data_pool])
        return data_map

    def knn(self):
        '''
        This class only have knn clustering method
        '''
        all_subsquences = []
        for i in self.data_map_train:
            single_time_series_part = i[1]
            for seq in single_time_series_part:
                all_subsquences.append(seq)
        print(len(all_subsquences))
        all_subsquences = np.array(all_subsquences)
        start_time = time.time()
        kmeans = KMeans(n_clusters=self.n_clustering, random_state=0, n_init="auto").fit(all_subsquences)
        end_time = time.time()
        print('Training KNN Time (sec): ' + str(end_time - start_time))
        return kmeans
    
    def build_clustering_occurrence(self, x, label):
        '''
        This data transformation process transfer the data based on each feature is occurring the time series instance ot not. Hence, the data do not include any time information based on each clustering feature and temporal relationship between different clustering features. 
        '''
        train_x = pd.DataFrame(columns=self.clustering_name,dtype=int)
        columns = {}
        for i in self.clustering_name:
            columns[i] = 0
        for item in x:
            columns = dict.fromkeys(columns, 0)
            all_c_index = [i.c_index for i in item]
            unique = set(all_c_index)
            for i in unique:
                columns[i] = 1
            train_x = train_x._append(columns, ignore_index=True)
        label = pd.DataFrame(label, columns=[self.label_name],dtype=int)
        all_data = pd.concat([train_x, label], axis=1)
        return all_data
        
    def generate_temporal_relations(self, format_rule_atom):
        '''
        This is the function to generate the temporal rules for each possible rules generated
        '''
        # number of predicate 
        for single_rule in format_rule_atom:    
            trainable_data = self.obtain_long_feature(single_rule, self.x_train, self.data_map_train)
            testing_data = self.obtain_long_feature(single_rule, self.x_test, self.data_map_test)
            rules, metrics  = self.ripper_learning(trainable_data, testing_data)
            print(rules)
            p,r = self.compute_precision_recall(rules, sub_rule=str(single_rule))
        
        return 0  

        
        
    
    def ripper_learning(self, train:pd.DataFrame, test:pd.DataFrame):
        '''
        train and test are in pandas format
        '''
        # shuffle train 
        train = train.sample(frac=1)
        y_train = train[self.label_name].astype(bool)
        x_train = train.drop(columns=[self.label_name])
        y_test = test[self.label_name].astype(bool)
        x_test = test.drop(columns=[self.label_name])
        
        drop_single_value_column = []
        for column in x_train.columns:
            if x_train[column].min() == x_train[column].max():
                drop_single_value_column.append(column)
                
        x_train = x_train.drop(columns=drop_single_value_column)
        # x_test = test.drop(columns=drop_single_value_column)
        
        x_train.to_csv('train.csv', index=False)
        
        print('Begin to train RIPPER')
        start_time = time.time()
        estimator = RipperExplainer()
        estimator.fit(x_train, y_train, target_label=bool(self.default_label))
        end_time = time.time()
        print('Training RIPPER Time (sec): ' + str(end_time - start_time))
        y_pred = estimator.predict(x_test)
        
        metrics_test = {}
        metrics_test['Accuracy'] = accuracy_score(y_test, y_pred)
        metrics_test['Precision'] = precision_score(y_test, y_pred, pos_label=self.default_label)
        metrics_test['Recall'] = recall_score(y_test, y_pred, pos_label=self.default_label)
        tn, fp, fn, tp =  confusion_matrix(y_test, y_pred).ravel()
        metrics_test['TP'] = tp
        metrics_test['FP'] = fp
        metrics_test['TN'] = tn
        metrics_test['FN'] = fn
        metrics_test['F1'] = metrics_test['Precision'] * metrics_test['Recall'] * 2 / (metrics_test['Precision'] + metrics_test['Recall'])
        metrics_test['TNR'] = tn / (tn + fp)
        
        rule_set = estimator.explain()
        
        print("Rules: \n", str(rule_set))
        
        return rule_set, metrics_test

    def DFOL_learning(self, train:pd.DataFrame, test:pd.DataFrame, minimum_precision=0.8, minimum_recall=0.8, maximum_iteration=10):
        dforl_model = dforl.RunModel(train=train, test=test, minimum_precision=minimum_precision, minimum_recall=minimum_recall, maximum_iteration=maximum_iteration, output_file_name=self.result_name+'/dforl.md', learn_rate=0.1,epoch=1000,batch_size=512, early_stopping=True, remove_low_precision=0.5,model_type='single')
        rule, metrics, _ =  dforl_model.run()
        return rule, metrics
        
class TimeSeriesNan(TimeSeries):
    def __init__(self, rule_learning_method='ripper',train_data_path = '', test_data_path = '', result_name='', task_name='', middle_data_path = '',past_windows_time = 24, unit = 1, low = None, high=None, clustering_method='knn',multi_class_label=None, n_clustering = 5) -> None:
        super().__init__(rule_learning_method,train_data_path, test_data_path, result_name, task_name, middle_data_path, past_windows_time, unit=unit, low = low, hig=high, clustering_method=clustering_method,multi_class_label=multi_class_label, n_clustering=n_clustering)
        
        self.X_hat = 0

    
    def knn(self, data=None, prev_centroids = None, max_iter=10):
        """Perform K-Means clustering on data with missing values.
        Args:
        X: An [n_samples, n_features] array of data to cluster.
        n_clusters: Number of clusters to form.
        max_iter: Maximum number of EM iterations to perform.

        Returns:
        labels: An [n_samples] vector of integer labels.
        centroids: An [n_clusters, n_features] array of cluster centroids.
        X_hat: Copy of X with the missing values filled in.
        """
        # Initialize missing values to their column means
        all_subsquences = []
        if data == None:
            data = self.data_map_train
        for i in data:
            single_time_series_part = i[1]
            for seq in single_time_series_part:
                all_subsquences.append(seq)

        X = np.array(all_subsquences)
        
        missing = ~np.isfinite(X)
        mu = np.nanmean(X, 0, keepdims=1)
        self.X_hat = np.where(missing, mu, X)


        for i in range(max_iter):
            if prev_centroids is not None: 
                cls = KMeans(self.n_clustering, init=prev_centroids)
            else:
                cls = KMeans(self.n_clustering)

            # if i > 0:
            #     # initialize KMeans with the previous set of centroids. this is much
            #     # faster and makes it easier to check convergence (since labels
            #     # won't be permuted on every iteration), but might be more prone to
            #     # getting stuck in local minima.
            #     cls = KMeans(self.n_clustering, init=prev_centroids)
            # else:
            #     # do multiple random initializations in parallel
            #     cls = KMeans(self.n_clustering)

            # perform clustering on the filled-in data
            labels = cls.fit_predict(self.X_hat)
            self.centroids = cls.cluster_centers_

            # fill in the missing values based on their cluster centroids
            self.X_hat[missing] = self.centroids[labels][missing]

            # when the labels have stopped changing then we have converged
            if i > 0 and np.all(labels == prev_labels):
                break

            prev_labels = labels
            prev_centroids = cls.cluster_centers_

        return {'kmeans':cls, 'center':self.centroids,'est': self.X_hat}
    
    def estimate_with_mean(self, data):
        all_subsquences = []

        for i in data:
            single_time_series_part = i[1]
            for seq in single_time_series_part:
                all_subsquences.append(seq)

        X = np.array(all_subsquences)
        
        missing = ~np.isfinite(X)
        mu = np.nanmean(X, 0, keepdims=1)
        X_hat = np.where(missing, mu, X)
        return X_hat
    
    def update_data_map(self):
        self.data_map_train = self.update_one_data_map(self.data_map_train, self.X_hat)
        
        # ! run knn to estimate the test data: one knn estimate, one mean estimate 
        # self.knn(self.data_map_test, self.centroids, max_iter=10)
        X_hat = self.estimate_with_mean(self.data_map_test)
        self.data_map_test = self.update_one_data_map(self.data_map_test, X_hat)
        
        
    def update_one_data_map(self, data, X_hat):
        data_new = []
        number_elements = len(data[0][1])
        ini_ele = 0
        for i in range(len(data)):
            label = data[i][0]
            single_data = []
            for j in range(number_elements):
                single_data.append(X_hat[ini_ele+j])
            ini_ele += number_elements
            data_new.append([label, single_data])
        return data_new
    

    
class TimeSeriesEvent(TimeSeries):
    def __init__(self, rule_learning_method='ripper', train_data_path='', test_data_path='', result_name='', task_name='', father_path='', unit=1, low=4, high=11,clustering_method = 'knn', max_plot = 50, min_precision=0.8, min_recall=0.8, max_inter=10,multi_class_label =None,n_clustering=5) -> None:
        super().__init__(rule_learning_method, train_data_path, test_data_path, result_name, task_name, father_path, unit, low, high, clustering_method=clustering_method, max_plot = max_plot, min_precision=min_precision, min_recall=min_recall, max_inter=max_inter,multi_class_label=multi_class_label,n_clustering=n_clustering)
        self.row_train, self.event_train = self.obtain_split_event_data(self.train_path)
        self.row_test, self.event_test = self.obtain_split_event_data(self.test_path)
        # self.points = [0,1,4,8,12,16,20,24,100]
        self.points = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]  
        self.threshold_dwt = 3 # control the similiarity between two subsequences 
        if self.body_windows_length==None:
            self.body_windows_length = len(self.row_train.columns)+ len(self.event_train.columns)-1
            # a = 1
        
    # redefine the split function to keep time series the clustering method, and keep the event data to as the features. like insulin-5 (medicine name-dosage). The relation with these event data and clustering data could be set as before1h() before2h etc. 
    # TODO: Try to propose a new method with neural predicate to learn more predicate. First implement the code on knowledge graph data, then move the work on the modelling for time series datasets. 
    
    def obtain_split_event_data(self,data_path):
        dp = pd.read_csv(data_path)
        all_column = dp.columns
        clustering_name = []
        event_mname = []
        clustering_name.append(self.label_name)
        for name in self.time_series_name:
            for column in all_column:
                if name in column:
                    clustering_name.append(column)
        for name in self.time_event_name:
            for column in all_column:
                if name in column:
                    event_mname.append(column)            
        clustering_data = dp[clustering_name]
        event_data = dp[event_mname]
        return clustering_data, event_data        
    

        
        
    def build_event_feature(self, event_data):
        '''
        This kind of events has values, which could be used as the features.
        '''
        event_features = [] 
        event_log = tqdm.tqdm(total=len(event_data),desc='Event Feature Generating',ascii=True)
        for index, row in event_data.iterrows():
            event_log.update(1)
            single_row_features = []
            for column in event_data.columns:

                # check row[column] is nan or not 
                if not np.isnan(event_data.at[index, column]):
                    time_point = int(column.split('-')[-1])
                    event_feature = column.split('-')[0]
                    event_index = int(event_feature.replace(self.time_event_name[0],''))
                    # start_time = event_data.columns.get_loc(column)
                    start_time = time_point
                    end_time = start_time
                    value = row[column]
                    
                    discrete_points = pd.cut([value], bins=self.points, right=False)
                    
                    range_value = str(discrete_points[0]).replace('(','from').replace(')','').replace(',','to').replace(' ','').replace('[','from').replace(']','')
                    c_index = f'E{event_index}{range_value}'
                    single_feature = Feature([],start_time=start_time, end_time=end_time, clustering_index=c_index)
                    single_row_features.append(single_feature)
            event_features.append(single_row_features)
        return event_features   
    
    def build_event_occurrence(self, event_data):
        single_data = {}
        for i in self.event_column_names:
            single_data[i] = 0
        occurrence_event_data = pd.DataFrame(columns=self.event_column_names,dtype=int)
        for item in event_data:
            single_data = dict.fromkeys(single_data, 0)
            all_c_index = [i.c_index for i in item]
            unique = set(all_c_index)
            for i in unique:
                single_data[i] = 1
            occurrence_event_data= occurrence_event_data._append(single_data, ignore_index=True)
        return occurrence_event_data
        
    def all_event_columns(self, event_data_train):
        all_event_features = set([])
        for rows in event_data_train:
            for index, item in enumerate(rows):
                all_event_features.add(item.c_index)
        return all_event_features
    
    

        
    def build_temporal_with_event(self, event_data, clustering_data, flag='Train'):
        '''
        Build the relational for clustering features and event features. The event features include the medicine type and dosage information. 
        '''

        all_pairs_clustering_event = itertools.product(self.clustering_name, self.event_column_names)
        all_pairs_event_clustering = itertools.product(self.event_column_names, self.clustering_name)
        all_temporal_atoms = {}
        for predicate in self.temporal_relation:
            for item in all_pairs_clustering_event:
                all_temporal_atoms[predicate+'_'+str(item[0])+'_'+str(item[1])+'_'] = 0
            for item in all_pairs_event_clustering:
                all_temporal_atoms[predicate+'_'+str(item[0])+'_'+str(item[1])+'_'] = 0
        # trainable_data_temporal = pd.DataFrame(columns=all_temporal_atoms.keys())
        trainable_data_temporal = []
        temporal_relation_bar = tqdm.tqdm(total=len(clustering_data),desc=f'Temporal Event Relation on {flag}',ascii=True)
        for index, item in enumerate(clustering_data):
            # set the initial ground truth for event-clustering predicate
            event_items = event_data[index]
            all_temporal_atoms = dict.fromkeys(all_temporal_atoms, 0)
            all_pairs_clustering_event = itertools.product(item, event_items)
            all_pairs_event_clustering = itertools.product(event_items, item)
            for c_pairs in all_pairs_clustering_event:
                clustering_index = c_pairs[0].c_index
                event_index = c_pairs[1].c_index
                clustering_end = c_pairs[0].end_time-1
                event_start = c_pairs[1].start_time
                for predicate, predicate_function in self.temporal_relation.items():
                    if predicate_function(clustering_end, event_start):
                        atom = predicate+'_'+str(clustering_index)+'_'+str(event_index)+'_'
                        all_temporal_atoms[atom] = 1
                    else:
                        atom = predicate+'_'+str(clustering_index)+'_'+str(event_index)+'_'
                        all_temporal_atoms[atom] = 0
            
            for c_pairs in all_pairs_event_clustering:
                event_index = c_pairs[0].c_index
                clustering_index = c_pairs[1].c_index
                event_start = c_pairs[0].start_time
                clustering_start = c_pairs[1].start_time
                for predicate, predicate_function in self.temporal_relation.items():
                    if predicate_function(event_start, clustering_start):
                        atom = predicate+'_'+str(event_index)+'_'+str(clustering_index)+'_'
                        all_temporal_atoms[atom] = 1
                    else:
                        atom = predicate+'_'+str(event_index)+'_'+str(clustering_index)+'_'
                        all_temporal_atoms[atom] = 0
            
            
            trainable_data_temporal.append(all_temporal_atoms)
            
            temporal_relation_bar.update(1)
        trainable_data_temporal = pd.DataFrame.from_dict(trainable_data_temporal)
        return trainable_data_temporal
        
    def data_process(self):
        if os.path.exists(os.path.join(self.father_path, 'data', self.data_split_senario)) == False:
            os.mkdir(os.path.join(self.father_path, 'data', self.data_split_senario))
        if os.path.exists(os.path.join(self.father_path, 'data', self.data_split_senario, 'event_feature_train.p')) and os.path.exists(os.path.join(self.father_path, 'data', self.data_split_senario, 'event_feature_test.p')):
            self.event_feature_train = pickle.load(open(os.path.join(self.father_path, 'data', self.data_split_senario, 'event_feature_train.p'), 'rb'))
            self.event_feature_test = pickle.load(open(os.path.join(self.father_path, 'data', self.data_split_senario, 'event_feature_test.p'), 'rb'))
        else:
            self.event_feature_train = self.build_event_feature(self.event_train)
            self.event_feature_test = self.build_event_feature(self.event_test)
            pickle.dump(self.event_feature_train, open(os.path.join(self.father_path, 'data', self.data_split_senario, 'event_feature_train.p'), 'wb'))
            pickle.dump(self.event_feature_test, open(os.path.join(self.father_path, 'data', self.data_split_senario, 'event_feature_test.p'), 'wb'))
        self.event_column_names_train = self.all_event_columns(self.event_feature_train)
        self.event_column_names_test = self.all_event_columns(self.event_feature_test)
        self.event_column_names = list(self.event_column_names_train.union(self.event_column_names_test))
        
        super().data_process(move_label=False)
        
        # ! key condition 
        if os.path.exists(os.path.join(self.father_path, 'data', self.data_split_senario, 'event_train.csv')) == False:
            train_pd_event = self.build_temporal_with_event(self.event_feature_train, self.x_aggregate_train,'Train')
            test_pd_event = self.build_temporal_with_event(self.event_feature_test, self.x_aggregate_test,'Test')
            occurrence_event_train = self.build_event_occurrence(self.event_feature_train)
            occurrence_event_test = self.build_event_occurrence(self.event_feature_test)
            self.train_pd = self.combine_train_pd(self.train_pd, train_pd_event)
            self.test_pd = self.combine_train_pd(self.test_pd, test_pd_event)
            self.train_pd = self.combine_train_pd(self.train_pd, occurrence_event_train)
            self.test_pd = self.combine_train_pd(self.test_pd, occurrence_event_test)
            
            de_log = tqdm.tqdm(total=len(self.train_pd.columns), desc='Delete Columns with only one value')
            delete_columns = []
            for col in self.train_pd.columns:
                if len(self.train_pd[col].unique()) == 1:
                    delete_columns.append(col)
                de_log.update(1)
            self.train_pd = self.train_pd.drop(columns=delete_columns)
            self.test_pd = self.test_pd.drop(columns=delete_columns)
            print(f'delete {len(delete_columns)} columns...')
            self.train_pd.to_csv(os.path.join(self.father_path, 'data', self.data_split_senario, 'event_train.csv'), index=False)
            self.test_pd.to_csv(os.path.join(self.father_path, 'data', self.data_split_senario, 'event_test.csv'), index=False)
        else:
            self.train_pd = pd.read_csv(os.path.join(self.father_path, 'data', self.data_split_senario, 'event_train.csv'))
            self.test_pd = pd.read_csv(os.path.join(self.father_path, 'data', self.data_split_senario, 'event_test.csv'))         

    def obtain_explainable_figure(self, format_rule_atom, cluster_data, estimated_data, train_event_data=None): 
        '''
        input: rules, test time series original data, knn data, 
        output: for each rule, generate a folder to store the explainable figure
        '''
        # if np.shape(cluster_data) == np.shape(self.x_train):
        #     train_event_data = self.event_feature_train
        # else:
        train_event_data = self.event_feature_test
        rule_highlight = defaultdict(list)
        rule_event_highlight = defaultdict(list)
        str_rules = []
        for rule in self.rules.conjunctions:
            str_rules.append(str(rule))
        for time_series_index, time_series_cluster in enumerate(cluster_data):
            unique_knn_cluster = np.unique(time_series_cluster)
            event_feature = [i.c_index for i in train_event_data[time_series_index]]
            event_time = [i.start_time for i in train_event_data[time_series_index]]
            # only check the positive label time series data
            if estimated_data[time_series_index][0] == 0:
                continue
            for rule_index,rule in enumerate(format_rule_atom):
                # if all positive literals in the rule 
                rule_glucose = rule.series
                rule_insulin = rule.event # each insulin should be the clustering class
                # TODO if satisfied the rule
                if set(rule_glucose).issubset(set(unique_knn_cluster)) and set(rule_insulin).issubset(set(event_feature)):
                    highlight_slices = []
                    for clustering_index, cluster in enumerate(time_series_cluster):
                        if cluster in rule_glucose:
                            atom_index = rule_glucose.index(cluster)
                            single_highlight_slice = single_highlights_slice(estimated_data[time_series_index][1][clustering_index], clustering_index, self.color[atom_index])
                            # highlight.append([estimated_data[time_series_index][1][clustering_index],clustering_index, self.color[atom_index]])
                            highlight_slices.append(single_highlight_slice)
                        else:
                            continue
                    highlight_event_data = []
                    for event_info in rule_insulin:
                        event_index = event_feature.index(event_info)
                        start_time = event_time[event_index]
                        highlight_event_data.append(Feature([],start_time, start_time, clustering_index=event_info))
                    rule_highlight_single = SingleRuleHighlights(highlight_slices, time_series_index, rule.precision, rule.recall, str_rules[rule_index])
                    rule_highlight_single.add_event_data(highlight_event_data)
                    rule_highlight[str(tuple(rule_glucose))].append(rule_highlight_single)
                    # rule_highlight[str(tuple(rule))].append([highlight, time_series_index, rule_p_r[1], rule_p_r[2], str_rules[rule_index]])
        return rule_highlight
    
    def plot(self, rule_highlight, test_data_path):
        # the first row is column name
        # if 'train' in test_data_path:
        #     test_data = self.row_train
        # else:
        test_data = self.row_test
        plot_path = self.result_name+'/rule_plot'
        if os.path.exists(plot_path) == False:
            os.makedirs(plot_path)

        rule_index = 0 
        plot_log = tqdm.tqdm(total=int(len(rule_highlight)*self.plot_max), desc='Plotting For Each Rule', ascii=True)
        for clustering_in_rule, estimated_value in rule_highlight.items():
            rule_path = os.path.join(plot_path, str(clustering_in_rule))
            if os.path.exists(rule_path) == False:
                if len(rule_path) > 300:
                    continue
                os.makedirs(rule_path)
            for data_index, all_plot in enumerate(estimated_value):
                plot_log.update(1)
                if data_index >= self.plot_max:
                    break
                time_series_index = all_plot.valid_timeseries_index
                time_series_raw = test_data.iloc[time_series_index][1:] # the first one is label
                fig = plt.figure(figsize=(20,6))
                ax = fig.add_subplot(111)
                time_series_raw = np.array(time_series_raw, dtype=float)
                total_length = self.body_windows_length + self.future_window_length
                plt.plot(range(0, self.body_windows_length, self.record_unit_time), time_series_raw, '-D', label=f'full time series {data_index}')
                for item in all_plot.single_high_slices:
                    start_index = item.start_index
                    sub_data = item.data
                    color_plot = item.color
                    plt.plot(range(start_index, start_index+self.subsequence_length, self.record_unit_time), sub_data, color_plot, alpha=0.5, linewidth=7.0)
                for item in all_plot.event_data: # event data should be the list of clustering event 
                    start_index = item.start_time
                    insulin_name = item.c_index
                    plt.scatter(start_index, 0 , color='green', s=100)
                    ax.text(start_index, 0, insulin_name, fontsize=12)
                title_str = ''
                list_key = clustering_in_rule.replace('(','').replace(')','').replace(' ','').split(',')
                for index,i in enumerate(list_key):
                    if 'E' in i:
                        continue
                    if ' ' == i:
                        continue
                    title_str +=  f'{self.color[index]}({i}) '
                ax.set_title(f'Rule: {all_plot.rule_str} \n {title_str}, p: {round(all_plot.precision,2)}, r: {round(all_plot.recall,2)}, \n No. {str(data_index)} time series data')
                    #  , first {self.color[0]}, second {self.color[1]}, third {self.color[2]}, fourth {self.color[3]}',fontsize=12)
                plt.xticks(np.arange(total_length+1),np.arange(self.body_windows_length,-1*self.future_window_length-1,-1), rotation=90)
                if self.low != None and self.high != None:
                    plt.hlines(y=self.low, xmin=0, xmax=total_length, linestyles='--', colors='k')
                    plt.hlines(y=self.high, xmin=0, xmax=total_length, linestyles='--', colors='k')
                plt.savefig(os.path.join(rule_path, str(data_index)+'.png'))
                plt.close()
            rule_index += 1
        return 0 
    
    
class TimeZoneSeries(TimeSeriesEvent):
    def __init__(self, rule_learning_method='ripper', train_data_path='', test_data_path='', result_name='', task_name='', father_path='', unit=1, low=4, high=11,clustering_method = 'knn', max_plot = 50, min_precision = 0.8, min_recall = 0.7, max_inter = 10,multi_class_label=None, n_clustering = 5) -> None:
        super().__init__(rule_learning_method, train_data_path, test_data_path, result_name, task_name, father_path, unit, low, high, clustering_method=clustering_method, max_plot = max_plot, min_precision=min_precision, min_recall=min_recall, max_inter=max_inter,multi_class_label=multi_class_label, n_clustering=n_clustering)

        self.total_window_length = self.body_windows_length + self.future_window_length
        self.single_zone_length = 12
        if self.total_window_length % self.single_zone_length != 0:
            self.largest_time_index = int(self.total_window_length/self.single_zone_length) + 1
        else:
            self.largest_time_index = int(self.total_window_length/self.single_zone_length)
        
        self.clustering_name = []
        for i in range(self.n_clustering):
            for j in range(self.largest_time_index):
                self.clustering_name.append('series'+str(i)+'at'+str(j))

    def self_temporal_features(self, data, log_info):
        all_pairs_clustering = itertools.product(self.clustering_name, self.clustering_name)
        all_temporal_atoms = {}
        for predicate in self.temporal_relation:
            for item in all_pairs_clustering:
                all_temporal_atoms[predicate+'_'+item[0]+'_'+item[1]] = 0
        all_atoms = set(all_temporal_atoms.keys())
        # trainable_data_temporal = pd.DataFrame(columns=all_temporal_atoms.keys())
        trainable_data_temporal = []
        temporal_relation_bar = tqdm.tqdm(total=len(data),desc=f'Temporal Relation on {log_info}',ascii=True)
        for item in data:
            all_temporal_atoms = dict.fromkeys(all_temporal_atoms, 0)
            all_pairs_clustering = itertools.permutations(item, 2)
            all_list = list(all_pairs_clustering)
            for c_pairs in all_list:
                first_index = c_pairs[0].c_index
                second_index = c_pairs[1].c_index
                first_end = c_pairs[0].end_time-1
                second_start = c_pairs[1].start_time
                first_time_zone = c_pairs[0].time_zones
                second_time_zone = c_pairs[1].time_zones
                for predicate, predicate_function in self.temporal_relation.items():
                    atom = f'{predicate}_{str(first_index)}{first_time_zone}_{str(second_index)}{second_time_zone}'
                    if atom in all_atoms:
                        if predicate_function(first_end, second_start):
                            all_temporal_atoms[atom] = 1
                        else:
                            all_temporal_atoms[atom] = 0
                    else:
                        raise ValueError('The atom is not in the all atoms')
            trainable_data_temporal.append(all_temporal_atoms)
            temporal_relation_bar.update(1)
        # if non_column == None:
        #     non_column = trainable_data_temporal.apply(pd.Series.nunique) != 1
        #     trainable_data_temporal = trainable_data_temporal.loc[:,non_column]   
        # else:
        #     trainable_data_temporal = trainable_data_temporal.loc[:,non_column]  
        trainable_data_temporal = pd.DataFrame.from_dict(trainable_data_temporal)
        return trainable_data_temporal
    
    def build_clustering_occurrence(self, x, label):
        '''
        This data transformation process transfer the data based on each feature is occurring the time series instance ot not. Hence, the data do not include any time information based on each clustering feature and temporal relationship between different clustering features. 
        '''
        
        train_x = pd.DataFrame(columns=self.clustering_name,dtype=int)
        columns = {}
        for i in self.clustering_name:
            columns[i] = 0
        for item in x:
            columns = dict.fromkeys(columns, 0)
            all_c_index = [(i.c_index,i.time_zones) for i in item]
            for i in all_c_index:
                atom = str(i[0])+str(i[1])
                columns[atom] = 1
            train_x = train_x._append(columns, ignore_index=True)
        label = pd.DataFrame(label, columns=[self.label_name],dtype=int)
        all_data = pd.concat([train_x, label], axis=1)
        return all_data
    
    def build_temporal_with_event(self, event_data, clustering_data, flag='Train'):
        '''
        Build the relational for clustering features and event features. The event features include the medicine type and dosage information. 
        '''

        all_pairs_clustering_event = itertools.product(self.clustering_name, self.event_column_names)
        all_pairs_event_clustering = itertools.product(self.event_column_names, self.clustering_name)
        all_temporal_atoms = {}
        for predicate in self.temporal_relation:
            for item in all_pairs_clustering_event:
                all_temporal_atoms[predicate+'_'+str(item[0])+'_'+str(item[1])] = 0
            for item in all_pairs_event_clustering:
                all_temporal_atoms[predicate+'_'+str(item[0])+'_'+str(item[1])] = 0
        # trainable_data_temporal = pd.DataFrame(columns=all_temporal_atoms.keys())
        trainable_data_temporal = []
        temporal_relation_bar = tqdm.tqdm(total=len(clustering_data),desc=f'Temporal Event Relation on {flag}',ascii=True)
        for index, item in enumerate(clustering_data):
            # set the initial ground truth for event-clustering predicate
            event_items = event_data[index]
            all_temporal_atoms = dict.fromkeys(all_temporal_atoms, 0)
            all_pairs_clustering_event = itertools.product(item, event_items)
            all_pairs_event_clustering = itertools.product(event_items, item)
            for c_pairs in all_pairs_clustering_event:
                clustering_index = c_pairs[0].c_index
                event_index = c_pairs[1].c_index
                clustering_end = c_pairs[0].end_time-1
                clustering_timezone = c_pairs[0].time_zones
                event_start = c_pairs[1].start_time
                for predicate, predicate_function in self.temporal_relation.items():
                    atom = f'{predicate}_{clustering_index}{clustering_timezone}_{event_index}'
                    if atom in all_temporal_atoms.keys():
                        if predicate_function(clustering_end, event_start):
                            all_temporal_atoms[atom] = 1
                        else:
                            all_temporal_atoms[atom] = 0
                    else:
                        raise ValueError('The atom is not in the all atoms')
            
            for c_pairs in all_pairs_event_clustering:
                event_index = c_pairs[0].c_index
                clustering_index = c_pairs[1].c_index
                event_start = c_pairs[0].start_time
                clustering_start = c_pairs[1].start_time
                clustering_timezone = c_pairs[1].time_zones
                for predicate, predicate_function in self.temporal_relation.items():
                    atom = f'{predicate}_{event_index}_{clustering_index}{clustering_timezone}'
                    if atom in all_temporal_atoms.keys():
                        if predicate_function(event_start, clustering_start):
                            all_temporal_atoms[atom] = 1
                        else:
                            all_temporal_atoms[atom] = 0
                    else:
                        raise ValueError('The atom is not in the all atoms')
            
            
            trainable_data_temporal.append(all_temporal_atoms)
            
            temporal_relation_bar.update(1)
        trainable_data_temporal = pd.DataFrame.from_dict(trainable_data_temporal)
        return trainable_data_temporal
    

    def transfer_rule_to_atom(self, rule_obj, p, r, l):
        '''
        Transfer the rules to atoms
        '''
        rules = str(rule_obj)
        all_rules = []
        rules_str = rules.replace('v','')
        rules_str = rules_str.split('\n')[1:-2]
        for rule_index, item in enumerate(rules_str):
            single_rule = RuleFormat()
            # ! append rule object 
            single_conjnunction = rule_obj.conjunctions[rule_index]
            single_rule.rule_single_obj = DnfRuleSet([single_conjnunction], then_part=True)
            items_conjunction = item.split('^')
            for i in items_conjunction:
                i = i.replace(' ','').replace('[','').replace(']','').replace('(','').replace(')','').replace(' ','')
                # only select positive literal in the rule
                if '==1' in i and 'before' not in i:
                    if 'series' in i:
                        name = i.split('==')[0]
                        name = name.replace('series','')
                        cluster_index = name.split('at')[0]
                        time_zone = name.split('at')[1]
                        atom_index = int(cluster_index)
                        time_zone = int(time_zone)
                        single_rule.update_rule({'series':(atom_index,time_zone)})
                    else:
                        end_index = i.find('=')
                        sub_string = i[:end_index]
                        single_rule.update_rule({'E':sub_string})
                elif 'before' in i and '==1' in i:
                    name = i.split('==')[0]
                    three_tuple = name.split('_')
                    for atom in three_tuple:
                        # ! here break point
                        if 'series' in atom:
                            name = atom.replace('series','').replace(' ','')
                            cluster_index = name.split('at')[0]
                            time_zone = name.split('at')[1]
                            atom_index = int(cluster_index)
                            time_zone = int(time_zone)
                            single_rule.update_rule({'series':(atom_index,time_zone)})
                        elif 'E' in atom:
                            single_rule.update_rule({'E':atom})
            single_rule.update_p_r(p[rule_index], r[rule_index],l[rule_index])
            all_rules.append(single_rule)
        return all_rules
    
    def obtain_explainable_figure(self, format_rule_atom, cluster_data, estimated_data, train_event_data=None): 
        '''
        input: rules, test time series original data, knn data, 
        output: for each rule, generate a folder to store the explainable figure
        '''
        # Change in this customized function 
        # if np.shape(cluster_data) == np.shape(self.x_train):
        #     train_event_data = self.event_feature_train
        #     cluster_data = self.x_aggregate_train
        # else:
        
        train_event_data = self.event_feature_test
        cluster_data = self.x_aggregate_test
        # begin to return highlight slices
        rule_highlight = defaultdict(list)
        str_rules = []
        for rule in self.rules.conjunctions:
            valid_atom = []
            for atom in rule.predicates:
                if atom.value==1:
                    valid_atom.append(atom)
            rule = Conjunction(valid_atom)
            str_rules.append(str(rule))
        for time_series_index, time_series_cluster in enumerate(cluster_data):
            # if time_series_index == 6:
                # a=  2
            unique_knn_cluster = []
            for i in time_series_cluster:
                index = int(i.c_index.replace('series',''))
                timezones = int(i.time_zones.replace('at',''))
                unique_knn_cluster.append((index,timezones))
            # each time series cluster include time and clustering index information
            event_feature = [i.c_index for i in train_event_data[time_series_index]]
            event_time = [i.start_time for i in train_event_data[time_series_index]]
            # only check the positive label time series data
            test_instance_label = estimated_data[time_series_index][0]
            # if estimated_data[time_series_index][0] == 0:
                # continue
            for rule_index,rule in enumerate(format_rule_atom):
                # if all positive literals in the rule 
                rule_series = rule.series
                rule_event = rule.event # each insulin should be the clustering class
                # ! append the data into rule, the event do not include time information now
                if set(rule_series).issubset(set(unique_knn_cluster)) and set(rule_event).issubset(set(event_feature)):
                    rule.update_series_data_index(time_series_index)
                    if test_instance_label == 0: # ! when plot, only plot positive label data
                        continue
                    highlight_slices = [] 
                    # enter each subsequence of the data
                    for clustering_index, cluster in enumerate(time_series_cluster):
                        index = int(cluster.c_index.replace('series',''))
                        time_zone = int(cluster.time_zones.replace('at',''))
                        single_cluster_index_time = (index,time_zone)
                        if single_cluster_index_time in rule_series:
                            atom_index = rule_series.index(single_cluster_index_time)
                            start_cluster = cluster.start_time
                            end_cluster = cluster.end_time
                            for data_index in range(start_cluster,end_cluster):
                                single_highlight_slice = single_highlights_slice(estimated_data[time_series_index][1][data_index], data_index, self.color[atom_index])
                                highlight_slices.append(single_highlight_slice)
                            # highlight.append([estimated_data[time_series_index][1][clustering_index],clustering_index, self.color[atom_index]])
                        else:
                            continue
                    highlight_event_data = []
                    for event_info in rule_event:
                        event_index = [i for i in range(len(event_feature)) if event_feature[i] == event_info]
                        start_time = [event_time[i] for i in event_index]
                        # event_index = event_feature.index(event_info)
                        # start_time = event_time[event_index]
                        for i in range(len(start_time)):
                            highlight_event_data.append(Feature([],start_time[i], start_time[i], clustering_index=event_info))
                        # highlight_event_data.append(Feature([],start_time, start_time, clustering_index=event_info))
                    rule_highlight_single = SingleRuleHighlights(highlight_slices, time_series_index, rule.precision, rule.recall, str_rules[rule_index])
                    rule_highlight_single.add_event_data(highlight_event_data)
                    key = ''
                    for item in rule_series:
                        key += f'{item[0]}at{item[1]},'
                    for item in rule_event:
                        key += f'{item},'
                    rule_highlight[key].append(rule_highlight_single)
                    # rule_highlight[str(tuple(rule))].append([highlight, time_series_index, rule_p_r[1], rule_p_r[2], str_rules[rule_index]])
        return rule_highlight

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

if __name__ == '__main__':
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
    
    
    # SGH Glucose and Insulin Task 
    father_path = 'rule_learning_original'
    
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
    # train_data_path = os.path.join(data_name, 'new_ins_glu_past24_target6_future_6_unit60_min_mrecord4_group_all_year_2018_abn_xnew_train.csvfull')
    # test_data_path = os.path.join(data_name, 'new_ins_glu_past24_target6_future_6_unit60_min_mrecord4_group_all_year_2019_abn_xnew_train.csvfull')
    # task_name = 'newxy_sgh_1819_abn_6'
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
    
    ### setting for UCR coffee ###
    # initial_time = datetime.now()
    # archive_name = 'Lightning2'
    # data_name = f'UCR/{archive_name}/'
    # train_data_path = os.path.join(data_name, f'{archive_name}_train.csv')
    # test_data_path = os.path.join(data_name, f'{archive_name}_test.csv')
    # task_name = f'ucr_{archive_name}'
    # # rule_learning_method = 'ripper'
    # rule_learning_method = 'DFORL'
    # result_name = os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')    
    # ts = TimeZoneSeries(rule_learning_method= rule_learning_method ,train_data_path=train_data_path, test_data_path=test_data_path, result_name=result_name, task_name = task_name, father_path=father_path, unit = 1, low=None, high=None, clustering_method='tsDWTmeans', max_plot=50, min_precision=0.8, min_recall=0.8, max_inter=10, n_clustering=10)
    # ts.run()
    # end_time = datetime.now()
    # runn_time = end_time - initial_time
    # with open('runningtime.md', 'a+') as f:
    #     print(f'{archive_name} running time is {runn_time}', file=f)
    # # ts.only_plot_rules()
    
    
    # ## setting for UCR
    # todo_task = get_all_unsovled_task()
    todo_task = get_all_task()
    focused_task = ['Lightning2','Coffee','ECG200','GunPoint','Italy']
    for task in todo_task:
        for subtask in focused_task:
            if 'Gun' in subtask and subtask == task:
                continue_flag = 0
                break
            elif subtask in task and 'Gun' not in subtask:
                continue_flag = 0
                break
            else:
                continue_flag = 1
        if continue_flag == 1:
            continue
        print(task)
        initial_time = datetime.now()
        data_name = f'UCR/binary_unsolved/{task}/'
        train_data_path = os.path.join(data_name, f'{task}_train.csv')
        test_data_path = os.path.join(data_name, f'{task}_test.csv')
        task_name = f'ucr_{task}'
        # rule_learning_method = 'ripper'
        rule_learning_method = 'DFORL'
        result_name = os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')    
        ts = TimeZoneSeries(rule_learning_method= rule_learning_method ,train_data_path=train_data_path, test_data_path=test_data_path, result_name=result_name, task_name = task_name, father_path=father_path, unit = 1, low=None, high=None, clustering_method='tsDWTmeans', max_plot=10, max_inter=1)
        
        # print all parameters in a file
        if os.path.exists(os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}')) == False:
            os.mkdir(os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}'))
        with open(os.path.join(father_path, 'res/' + task_name + f'_{rule_learning_method}', 'parameters.md'), 'a+') as f:
            f.write(f'**{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}**\n')
            print(str(ts.__dict__), file=f)
        ts.run()
        end_time = datetime.now()
        runn_time = end_time - initial_time
        with open('runningtime.md', 'a+') as f:
            print(f'{task} running time is {runn_time}', file=f)
        add_task_to_solved_folder(task)
        
        
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