# -*- coding: utf-8 -*-
"""AprioriAlgoithm_Pyspark.ipynb

"""

sc.stop()
from pyspark import SparkContext, SparkConf, SQLContext 

# #Spark Config
conf = SparkConf().setAppName("sample_app")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)





from google.colab import files

uploaded = files.upload()

data1 = 'FP_Growth_textfile_dataset_Final with Single line.txt'

rdd = sc.parallelize(data1)

Rdd1 = rdd.map(lambda x: (x,1))

Rdd1.collect()

# imports
from collections import defaultdict
from itertools import chain, combinations


# class definition
class Apriori:

    def __init__(self):
        """
            declaring and initializing all the variables with default values
        """
        # variables
        self.dataset = []
        # all L values with their frequency in the dataset separated by k values
        self.all_frequent_item_with_frequency = {}
        # all L values with their frequency in a single dictionary
        self.aggregate_frequent_item_with_frequency = dict()
        # all candidate itemsets in a single dictionary
        self.aggregate_candidate_item_with_frequency = dict()
        # support score of itemsets along with the itemsets having minimum support score
        self.support_of_itemsets = []
        # association rules having the minimum confidence score
        self.association_rules = []
        # set of all the items
        self.item_set = set()
        self.minimum_support_threshold = 0
        self.minimum_confidence_threshold = 0
        self.intended_meaning=''

    def read_file(self, input_file_name):
        """
            reading the input file for transactions
        """
        input_file = open(input_file_name, 'r')
        lines = input_file.readlines()
        input_file.close()
        for line in lines:
            words_split = line.split(',')
            words_split_processed = list(map(lambda s: s.strip(), words_split))
            self.intended_meaning=words_split_processed[-1]
            words_split_processed.sort()
            self.dataset.append(words_split_processed)

    def print_output(self):
        """
            prints all the frequent items along with the association rules
        """
        print('***************************************************************')
        print('Support Scores of Selected Items')
        print('Total Items: {}'.format(len(self.support_of_itemsets)))
        print('***************************************************************')
        print('---------------------------------------------------------------')
        print('{0:30}{1:^20}'.format('Item Set', 'Support Score in Percentage'))
        print('---------------------------------------------------------------')
        frequent_items = []
        for itemset in self.support_of_itemsets:
            print('{:<25}{:>25.02f}%'.format(str(sorted(itemset[0])), itemset[1] * 100))
            print('---------------------------------------------------------------')

        print('***************************************************************')
        print('***************************************************************************')
        print('Association Rules')
        print('Total Association Rules: {}'.format(len(self.association_rules)))
        print('***************************************************************************')
        print('---------------------------------------------------------------------------')
        print('{0:<25}{1:^20}{2:^15}'.format('X', 'Y', 'Confidence Score in Percentage'))
        print('---------------------------------------------------------------------------')
        for rule in self.association_rules:
            print(
                '{0:<25} > {1:<20}{2:>15.2f}%'.format(str(sorted(list(rule[0]))), str(sorted(list(rule[1]))), rule[2]))
            print('---------------------------------------------------------------------------')
        print('***************************************************************************')

    def generate_item_set(self):
        """
            generates set of items
        """
        for data in self.dataset:
            for item in data:
                self.item_set.add(frozenset([item]))

    def subsets(self, itemset):
        """
            creates subset from an itemset
        """
        subset_list = list(chain.from_iterable(combinations(itemset, r) for r in range(len(itemset) + 1)))
        # removing empty tuple from the subset_list
        subset_list_filtered = list(filter(lambda subset: len(list(subset)) != 0, subset_list))
        # converting each tuple into frozenset
        subset_list_processed = [frozenset(subset) for subset in subset_list_filtered]
        return subset_list_processed

    def get_current_candidate_item_frequency(self, current_candidate_items):
        """
            generates a dictionary current candidate items with their frequency
        """
        current_candidate_item_frequency = defaultdict(int)
        # generating the current candidate frequency dictionary
        for candidate in current_candidate_items:
            for data in self.dataset:
                if candidate.issubset(set(data)):
                    current_candidate_item_frequency[candidate] = current_candidate_item_frequency[
                                                                      candidate] + 1
        # updating the aggregate candidate item frequency dictionary
        self.update_aggregate_candidate_item_frequency_dictionary(current_candidate_item_frequency)
        return current_candidate_item_frequency

    def get_current_frequency_set(self, current_candidate_item_with_frequency):
        """
            generates the current frequency set from the current candidate set
        """
        current_candidate_item_with_frequency = self.get_current_candidate_item_frequency(
            current_candidate_item_with_frequency)
        frequent_item_frequency_dictionary = defaultdict(int)
        # generating the frequent item frequency dictionary from the generated
        # current candidate frequency dictionary
        for candidate in current_candidate_item_with_frequency:
            if self.minimum_support_threshold <= current_candidate_item_with_frequency[candidate]:
                frequent_item_frequency_dictionary[candidate] = current_candidate_item_with_frequency[candidate]
        # updating the aggregate frequent item frequency dictionary
        self.update_aggregate_frequent_item_frequency_dictionary(frequent_item_frequency_dictionary)

        return current_candidate_item_with_frequency, frequent_item_frequency_dictionary

    def get_support_of_an_item(self, itemset):
        """
            generating the support score of each itemset
        """
        return float(self.aggregate_candidate_item_with_frequency[itemset]) / len(self.dataset)

    def join_dictionary(self, frequent_item_frequency_dictionary, k):
        """
            creating a dictionary having k length elements by joining current frequent items
            with itself to get the next candidate items
         """
        frequent_item_frequency_keys = frequent_item_frequency_dictionary.keys()
        frequent_item_frequency_keys_joined = []
        for i in frequent_item_frequency_keys:
            for j in frequent_item_frequency_keys:
                if len(i.union(j)) == k:
                    frequent_item_frequency_keys_joined.append(i.union(j))
        frequent_item_frequency_frozenset_key_joined_unique = set(frequent_item_frequency_keys_joined)
        return frequent_item_frequency_frozenset_key_joined_unique

    def update_aggregate_frequent_item_frequency_dictionary(self, frequent_item_frequency_dictionary):
        """
            updating the aggregate frequent item frequency dictionary
        """
        self.aggregate_frequent_item_with_frequency.update(dict(frequent_item_frequency_dictionary))

    def update_aggregate_candidate_item_frequency_dictionary(self, current_candidate_frequency_dictionary):
        """
            updating the aggregate candidate item frequency dictionary
        """
        self.aggregate_candidate_item_with_frequency.update(dict(current_candidate_frequency_dictionary))

    def generate_support_score_of_itemsets_having_minimum_support_score(self):
        """
            generating a list of itemsets having minimum support score with their support scores
         """
        support_score_of_itemsets_having_minimum_support_score = []
        for itemset in self.aggregate_frequent_item_with_frequency:
            support_score_of_itemsets_having_minimum_support_score.append(
                [list(itemset), self.get_support_of_an_item(itemset), itemset])
        return support_score_of_itemsets_having_minimum_support_score

    def generate_association_rules(self):
        """
            generating the association rules having the minimum confidence score
        """
        association_rules = []
        for key in self.all_frequent_item_with_frequency:
            for itemset in self.all_frequent_item_with_frequency[key]:
                if len(itemset) > 1:
                    subsets = self.subsets(itemset)
                    for subset in subsets:
                        difference = itemset.difference(subset)
                        if len(difference) > 0:
                            confidence_score_of_the_rule = (self.get_support_of_an_item(
                                itemset) / self.get_support_of_an_item(subset)) * 100

                            if confidence_score_of_the_rule >= self.minimum_confidence_threshold:
                                association_rules.append([subset, difference, confidence_score_of_the_rule])
        return association_rules

    def apriori(self, minimum_support_threshold, minimum_confidence_threshold):
        """
            executing the apriori algorithm
        """
        # setting the minimum support threshold
        self.minimum_support_threshold = minimum_support_threshold
        # setting the minimum confidence threshold
        self.minimum_confidence_threshold = minimum_confidence_threshold
        # generating frequency of items
        self.generate_item_set()
        # initializing the first stage candidate set
        current_candidate_set = self.item_set
        # k=1 is the length of items
        current_candidate_frequency_dictionary, frequent_item_frequency_dictionary = self.get_current_frequency_set(
            current_candidate_set)

        # length of items
        k = 2
        while (len(frequent_item_frequency_dictionary) != 0):
            # storing each frequent item frequency dictionary
            self.all_frequent_item_with_frequency[k - 1] = frequent_item_frequency_dictionary
            # joining current set of frequent items with itself
            # to get the next set of frequent items
            frequent_item_frequency_dictionary = self.join_dictionary(frequent_item_frequency_dictionary, k)
            # generating the current candidate frequency dictionary
            # along with the frequent item frequency dictionary
            current_candidate_frequency_dictionary, frequent_item_frequency_dictionary = self.get_current_frequency_set(
                frequent_item_frequency_dictionary)
            k = k + 1
        # generating support score of itemsets along with the
        # itemsets having minimum support score
        self.support_of_itemsets = self.generate_support_score_of_itemsets_having_minimum_support_score()
        # generating association rules having the minimum confidence score
        self.association_rules = self.generate_association_rules()
        return self.support_of_itemsets, self.association_rules, self.intended_meaning

def filter_association_rules(association_rules,intended_meaning):
    filtered_association_rules=[]
    for rule in association_rules:
        if rule[1]==frozenset([intended_meaning]):
            filtered_association_rules.append(rule)
    return filtered_association_rules

def test_sentences(test_file_name,association_rules):
    testing_dataset=[]
    test_file = open(test_file_name, 'r')
    lines = test_file.readlines()
    test_file.close()
    for line in lines:
        words_split = line.split(' ')
        words_split_processed = list(map(lambda s: s.strip(), words_split))
        words_split_processed.sort()
        testing_dataset.append(words_split_processed)
    count=1
    for words in testing_dataset:
        print('Sentence Number: ',count)
        count=count+1
        print(words)
        print('-----------------------------')
        subset_list = list(chain.from_iterable(combinations(words, r) for r in range(len(words) + 1)))
        # removing empty tuple from the subset_list
        subset_list_filtered = list(filter(lambda subset: len(list(subset)) != 0, subset_list))
        filtered_list=[list(x) for x in subset_list_filtered]
        for subset in filtered_list:
            for rule in association_rules:
                r_0=sorted(list(rule[0]))
                r_1=sorted(list(rule[1]))
                if subset==r_0 and len(subset)>1 and ('কথা' in subset) and len(r_1)==1  and rule[2]>90 :
                    print(r_0,'>',r_1,rule[2])
        print('-----------------------------')

# starting point of the program

if __name__ == '__main__':
    # name of the file to read
    input_file_name = sqlContext.load(Rdd1)
    test_file_name= sqlContext.load(Rdd1)
    # minimum support threshold
    minimum_support_threshold = 3
    # minimum confidence threshold
    minimum_confidence_threshold = 90
    # creating the Apriori object
    apriori = Apriori()
    # reading data from the file
    apriori.read_file(input_file_name)
    # executing the apriori algorithm
    print('##########################################################################################')
    print('Training Phase')
    print('##########################################################################################')
    apriori.apriori(minimum_support_threshold, minimum_confidence_threshold)
    # print the output of the apriori algorithm
    apriori.print_output()
    
    final_association_rules=filter_association_rules(apriori.association_rules,apriori.intended_meaning)
    
    print('***************************************************************')
    print('***************************************************************************')
    print('Final Filtered Association Rules')
    print('Total Association Rules: {}'.format(len(final_association_rules)))
    print('***************************************************************************')
    print('---------------------------------------------------------------------------')
    print('{0:<25}{1:^20}{2:^15}'.format('X', 'Y', 'Confidence Score in Percentage'))
    print('---------------------------------------------------------------------------')
    for rule in final_association_rules:
        print('{0:<25} > {1:<20}{2:>15.2f}%'.format(str(list(rule[0])), str(list(rule[1])), rule[2]))
        print('---------------------------------------------------------------------------')
    print('***************************************************************************')
    print('##########################################################################################')
    print('Testing Phase')
    print('##########################################################################################')
    test_sentences(test_file_name,final_association_rules)

