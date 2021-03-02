# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:28:42 2021

@author: ernst
"""
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import random
import string
import hmac
import csv
import typing

class Helper_Methods:
    
    def __init__(self, dataSet = {}):
        self.dataSet = dataSet
        
    @property
    def dataSet(self):
        return self._dataSet
    
    @dataSet.setter
    def dataSet(self, newSet):
        self._dataSet = newSet
    


#Get Game Info (made by Minding the Data)
#-----------------------------------------------------------------------------#
    def get_result(self, game_hash: str, salt: str) -> int:
        
        '''
        This method returns the score from a game based on the hash code.
        '''
        
        hm = hmac.new(str.encode(game_hash), b'', hashlib.sha256)
        hm.update(salt.encode("utf-8"))
        h = hm.hexdigest()
        if (int(h, 16) % 33 == 0):
            return 1
        h = int(h[:13], 16)
        e = 2**52
        return (((100 * e - h) / (e-h)) // 1) / 100.0
    
    def get_prev_game(self, hash_code: str) -> str:
        
        '''
        This method will return the previous game hash code.
        '''
        
        m = hashlib.sha256()
        m.update(hash_code.encode("utf-8"))
        return m.hexdigest()
    
#-----------------------------------------------------------------------------#
#Build dataset with no transformations
#-----------------------------------------------------------------------------#
    def build(self, game_hash: str, salt: str, num_variables: int) -> list:
        
        '''
        This method will return all of the game results from game_hash to num_variables
        games, capped at the total games that have occurred.
        '''
        
    #Collect All Game Results
        first_game = "77b271fe12fca03c618f63dfb79d4105726ba9d4a25bb3f1964e435ccf9cb209"
        results = []
        count = 0
        
        while count < num_variables:
            count += 1
            results.append(self.get_result(game_hash, salt))
            game_hash = self.get_prev_game(game_hash)
            
        self.dataSet = results
    
#-----------------------------------------------------------------------------#
#Build Dataset features
#-----------------------------------------------------------------------------#
    #dataset, 1.2, 1.5, 2, 5, 10
    def build_tier(self,
                   cutoff1: int, cutoff2: int, cutoff3: int, cutoff4: int
                   ,cutoff5: int) -> list:
        
        '''
        This method builds the tier variable, which is included
        in the x matrix of features.  It is a range between 0-5 that 
        gives some insight into how high is projected.
        '''
        
        yList = []
        
        for a in range(len(self.dataSet)):
            if(self.dataSet[a] <= cutoff1):
                yList.append(0.0)
            elif(self.dataSet[a] <= cutoff2):
                yList.append(1.0)
            elif(self.dataSet[a] <= cutoff3):
                yList.append(2.0)
            elif(self.dataSet[a] <= cutoff4):
                yList.append(3.0)
            elif(self.dataSet[a] <= cutoff5):
                yList.append(4.0)
            else:
                yList.append(5.0)
            
        self.dataSet = yList
    
    def build_tiers(self, data: list, cutoff1: int,
                   cutoff2: int, cutoff3: int,
                   cutoff4: int,cutoff5: int) -> list:
        '''function is used for a wacky normalization concept I tried.'''
        h = Helper_Methods()
        h.dataSet = data
        h.build_tier(cutoff1, cutoff2, cutoff3, cutoff4, cutoff5)
        return h.dataSet
      
        
    def randomDice(self):
        '''used to automate a monotonous task in my geology class, was too lazy
        to load a fresh file to do it.'''
        count = []
        init = 50
        
        for a in range(0,20):
            valc = []
            for a in range(init):
                val = random.randint(1, 6)
                valc.append(val)
            counter = 0
            
            for a in valc:
                if(a==1):
                    counter += 1
            count.append(counter)
            
            init -= counter
            
        return count
    
    def createOffset(self) -> list:
        '''creates an offset between x & y.  This is because you don't want 
        x to be directly correlated with y, so there needs to be a 1 score
        lag between them (x corresponds to next y result)'''
        x_list = self.dataSet[:-1]
        self.dataSet = self.dataSet[1:]
        return x_list        
                
          
    def setMaxValue(self, maxValue) -> list:
        
        '''
        This normalizes the data to be below a certain value to remove outliers
        and improve accuracy.
        '''
        
        for index, num in enumerate(self.dataSet):
            if(num > maxValue):
                self.dataSet[index] = maxValue
                
    
    
    def getMax(self) -> int:
        
        '''
        Helper method primarily utilized in local methods.
        '''
        
        max: float = 0.0
        
        for _, num in enumerate(self.dataSet):
            if(num > max):
                max = num     
        return max
    
    def normalize(self):
        
        '''
        this function will normalize all of the data between 0 and 1.
        '''

        max: int = self.getMax()
        
        for index, num in enumerate(self.dataSet):
            num = num/max
            self.dataSet[index] = num
    
    def reverse_normalization(self, max: int) -> list:
        '''
        returns a list of values that are no longer normalized
        between 0 and 1
        '''
        for index, num in enumerate(self.dataSet):
            num = float(num * max)
            self.dataSet[index] = num
        

    
    def high_values(self, cutoff: int) -> list:
        '''normalize between high scores and low scores (1,0)'''
        results = []
        for _, value in enumerate(self.dataSet):
            if(value >= cutoff):
                results.append(1)
            else:
                results.append(0)
                
        return results
        
    
    def find_consecutive_red(myList: list, howMany: int) -> list:
        '''find consecutive losses'''
        results = []
        for index, value in enumerate(myList):
            hasRun = False
            
            #if index or value are less than min, score is 0
            if(index < howMany or value > 1.99):
                results.append(0)
                continue
            
            for a in range(1, howMany):
                if(myList[index - a] > 1.99):
                    results.append(0)
                    hasRun = True
                    break
                
            if(hasRun == False):
                results.append(1)
            
        return results
    
    def find_outliers(self, max: int, x_list: list) -> list:
        '''find outlier values based on the optimal max value (determined with other methods)'''
        results = []
        
        for index, value in enumerate(x_list):
            if(value == max):
                results.append(1)
            else:
                results.append(0)
                
        return results
    
    def find_most_consecutive(self) -> int:
    '''find most consecutive wins or losses, can be determined by changing the > to < in the while loop.'''
        for index, value in enumerate(self.dataSet):
        
            max: int = 0
            
            for index, value in enumerate(self.dataSet):
                if(index < 20):
                    continue
                
                if(value > 1.99):
                    count: int = 1
                    total: int = 1
                    while(self.dataSet[index - count] > 1.99):
                        count += 1
                        total += 1
                    
                    if(total > max):
                        max = total
            
            return max
        
    def split_Y(self, dataset: list, ratio: float) -> list:
        '''split independent variable vector into training and testing sets'''
        index: int = len(dataset) * ratio
        index = int(index)
        testAndTraining = [] #0: training, 1: test
        training: list = dataset[0:index]
        test: list = dataset[index:]
        testAndTraining.append(training)
        testAndTraining.append(test)
        return testAndTraining
    
    def check_last(self, data, score) -> int:
        if(data[1] > score):
            return 0
        else:
            return 1
        
    def increase_decrease(self) -> list:
        '''normalize the dataset into increasing next round and decreasing next round (1,0, respectively)'''
        results = list()
        for index, value in enumerate(self.dataSet):
            if(index == len(self.dataSet)-1 or self.dataSet[index+1] > value):
                results.append(1)
            else:
                results.append(0)
            if(index == len(self.dataSet)-1):
                return results
                
    def find_difference(self):
        '''find difference in times the score is higher next round than lower next round,
        based on the last round.'''
        inc = 0
        dec = 0
        for a in self.dataSet:
            if(a == 1):
                inc+=1
            else:
                dec += 1
        print(inc - dec)
                
    
    def classify(self, my_list: list) -> list:
        '''splits between red/green games.'''
        
        classified = []
        
        for a in my_list:
            if(a >= 2.00):
                classified.append(1.0)
            else:
                classified.append(0)
        return classified
    
    def split_matrix_features(self, matrix: list, ratio: int) -> list:
        '''split matrix of features into training and testing sets.'''
        training_set: list = []
        testing_set: list = []
        
        #Testing/training cutoff
        index: int = int(len(matrix[1]) * ratio)
        
        #Go through all features besides the dependent variable vector
        for _, vector in enumerate(matrix[1:]):
            temptrain: list = vector[:index]
            temptest: list = vector[index:]
            
            #Append to sets that will be returned
            testing_set.append(temptest)
            training_set.append(temptrain)
            
        return_set = [training_set, testing_set]
        return return_set
    
    def verify_accuracy(self, predicted: list, true: list):
        '''verify model accuracy.'''
        count = 0
        acc = 0
        
        for index, value in enumerate(predicted):
            print(true[index], " ", value)
            if value == true[index]:
                acc += 1
            count+= 1
        print(acc)
        print(count)
        return float(acc/count)
        
#-----------------------------------------------------------------------------#
#Probability-Related
#-----------------------------------------------------------------------------#
    def return_probability(self, limit: int, greaterThan: bool) -> float:
        '''finds proabbility that a number will be greater than a certain value.
        Used to find the optimal probability to begin classifying values as outliers.'''
        totalValCount: int = len(self.dataSet)
        targets: int = 0
        
        if(greaterThan == True):
            for _, value in enumerate(self.dataSet):
                if(value > limit):
                    targets += 1
        else:
            for _, value in enumerate(self.dataSet):
                if(value < limit):
                    targets ++ 1
        
        return targets/totalValCount
    
    def find_ideal_probability(self)-> float:
        '''
        This method finds the ideal number to set as max by finding a precise value
        where numbers start to become outliers.
        '''
        max: int = self.getMax()
        
        #loop through values ranging from 5-> max with step 2 to save
        #computation time, find the point where probability to be greater than 
        #is below 0.05.
        for a in range(5, int(max), 2):
            if(self.return_probability(a, True) <= 0.1):
                return a-1;
            
    def list_np_array(self, myList: list):
        '''turns a list into a np array.'''
        
        newList = np.array(myList)
            
        return newList
    
    def list_of_lists(self, data: list) -> list:
        '''turns a list into a list of lists.'''
        newList = []
        
        for a in data:
            n = list()
            n.append(a)
            newList.append(n)
        return newList
    
    
    def extend_list(self, singleList: list, bigList: list) -> list:
        
        for index, value in enumerate(singleList):
            bigList[index].append(value)
            
        
        return bigList
    
    def find_chance_consecutive(self) -> list:
        
        '''
        Finds the probability that there is a consecutive amount of 
        reds in a row, from 2 to the all-time max of 16.
        '''
        
        results = []
        
        #For each consecutive amount from 2-16
        for a in range(2, 16):
            #Create dataset that marks reds in a row with this cutoff
            res = Helper_Methods.find_consecutive_red(self.dataSet, a)
            
            #dings keeps track of each instance of reds in a row, count is the index.
            dings= 0
            count = 0
            while(count < len(res)):
                if(res[count] == 1):
                    #Keeps it from factoring in each 1 in the dataset (otherwise 16 in a row would count for 16 dings!)
                    while(res[count == 1]):
                        count += 1
                    dings += 1
                count += 1
            results.append([a, dings])
            
        return results
        
        
        
    def find_consecutive_green(self, howMany: int) -> list:
        '''find consecutive wins'''
        results = []
        for index, value in enumerate(self.dataSet):
            
            hasRun = False
            #if index or value are less than min, score is 0
            if(index < howMany or value < 1.99):
                results.append(0)
                continue
            
            for a in range(1, howMany):
                if(self.dataSet[index - a] < 1.99):
                    results.append(0)
                    hasRun = True
                    break #breaks this inner for loop
                
            if(hasRun == False):
                results.append(1)
        return results
        
            
        
            
        
#-----------------------------------------------------------------------------#
#Graphing
#-----------------------------------------------------------------------------#    

    def visualize(results):
        import seaborn as sns
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        plt.hist(results, range=(0, 25))
        plt.title("Histogram of Game Results", fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel("Number of Games", fontsize=15)
        plt.xlabel("Multiplier", fontsize=15)
        
