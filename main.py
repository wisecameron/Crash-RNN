'''This program will use a Recurrent Neural Network to make a judgement on the
outcome of the next crash round alongside a probability score. 


Writeup: 
    This algorithm can be used in conjunction with helper_methods to test a variety
    of different strategies in the popular online gambling game crash.
    You will find that no strategy is actually profitable as per the kelly criterion,
    but I did find the optimal strategy to maximize the chance of earning money,
    which can be found on my website, fromhomegigs.com
    
    The Helper_methods class can be used to tune the recurrent neural network
    found in CrashRNN.py to test a variety of different strategies.
    
    HOW IT WORKS:
        Each round provides a hash code that can be used to find the results of the round.
        You can generate previous games with this hash code.
        I used the hash code to generate num_variables games, compiled them into
        a np.array, then ran various methods to build a matrix of features
        which analyzed the data from multiple standpoints to determine
        if there were any correlations between round results which could be
        used by a sequential model to determine the probable results of the next 
        round.
    
    As you can see, the current state of the model is not really significant.
    Its main purpose is to be tuned using the paramaters specified below
    to test various strategies and gain insight into the game.
    
            
            BUGS:
                num_variables must be above 90
            

'''

import Helper_Methods
import RunOperation
import sys
import CrashRNN
import typing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard as tb
from keras.models import load_model

salt: str = "0000000000000000000fa3b65e43e4240d71762a5bf397d5304b2596d116859c"
game_hash: str = '9914966bf0f7f80fb9bca2180625b1731969cb584c80dd0d133a315996eb3262'
num_variables: int = 1000000
num_features = 2

class main():
    
    def main():
        
        while(True):
            try:
                train = input("Train model? Y/N: ").upper()
                break
            except(ValueError):
                print("Retry, incorrect input.")


        if(train == 'Y'):
            game_hash = '9914966bf0f7f80fb9bca2180625b1731969cb584c80dd0d133a315996eb3262'
            #Create two separate objects for storing data and using helper methods
            oldSet = Helper_Methods.Helper_Methods()
            yBuild = Helper_Methods.Helper_Methods()
            #Create identical datasets.
            oldSet.build(game_hash, salt, num_variables)
            yBuild.dataSet = oldSet.dataSet.copy()
            
            #Find ideal maximum value to stop outliers, normalize outliers around max
            yBuild.dataSet = yBuild.increase_decrease()
            
            #Create offset -- y_list is FUTURE, x_list is using LAST ROUND TO PREDICT FUTURE -- STILL Y VALUES!!!
            x_list = yBuild.createOffset()
            maintained_x = (oldSet.normalize())
            maintained_x = oldSet.createOffset()
            del(maintained_x[-1])
            print(maintained_x)
            print(len(maintained_x) == len(x_list))
            #Create list for y values, normalize them between 1/0 (1 == >= 2.0)
            yData = yBuild.dataSet
    
            #----------------------------------------------------------------------#
            #Create Matrix of Features -> x/yData :::: list of NP ARRAY
            #----------------------------------------------------------------------#     
            #Oldset dataset gets x_list -> used to create matrix of features!
            oldSet.dataSet = x_list
    
            #Create Matrix of Features dataset -- concatenate values into a list of lists,
            #Grouping them sequentially in nested lists by index
            xData = yBuild.list_of_lists(x_list)
            xData = yBuild.extend_list(maintained_x, xData)
            xData = np.array(xData)
            xData = xData.reshape(1, len(xData), num_features)
            yData = np.array(yData)
            yData = yData.reshape(1, len(yData), 1)
            
            
            #----------------------------------------------------------------------#
            #Split training and testing data sets -- > 80:20 split
            #----------------------------------------------------------------------#
            
            #Split training and testings sets
            x_train = xData[0][:int(len(xData[0])*0.8)]
            x_test = xData[0][int(len(xData[0])*.8):]
            y_train = yData[0][:int(len(yData[0])*.8)]
            y_test = yData[0][int(len(yData[0])*.8):]
        
            
            #Reshape data, pass into rnn object -> n_samples, timesteps, features
            x_train = x_train.reshape(len(x_train), 1, num_features)
            x_test = x_test.reshape(len(x_test), 1, num_features)
            y_train = y_train.reshape(len(y_train), 1)
            y_test = y_test.reshape(len(y_test), 1)
            
            #RNN = CrashRNN.CrashRNN(x_train, x_test, y_train, y_test)
            #RNN.create_model()
            
        else:
            print("INITIALIZING MODEL...")
            model = load_model("fixed_model")
            h = Helper_Methods.Helper_Methods()
            
            print("MODEL INITIALIZED.")
            print("CREATING INITIAL DATASET, PLEASE SPECIFY GAME_HASH")
            firstRun = True
            increase = []
            vals = []
            
            while(True):
                if(firstRun):
                    #Build dataset
                    game_hash = input('ENTER GAME HASH: ')
                    build = Helper_Methods.Helper_Methods()
                    build.build(game_hash, salt, num_variables)
                    
                    #Process Data
                    build.dataSet = build.increase_decrease()
                    x_list = build.dataSet
                    
                    vals.append(build.dataSet[0])
                    x_list = build.list_of_lists(x_list)
                    x_list = np.array(x_list)
                    x_list = x_list.reshape(len(x_list), 1, 1)
                    
                    #Make prediction
                    predict = model.predict(x_list)
                    predict = np.argmax(predict, axis = 1)
                    print(predict[0])
                    firstRun= False
                    
                else:
                    lastScore: int = input("Enter previous score: ")
                    vals.append(lastScore)
                    lastScore = h.check_last(vals, lastScore)
                    x_list = np.insert(x_list, 0, lastScore)
                    x_list = x_list.reshape(len(x_list),1,1)
                    predict = model.predict(x_list)
                    print(np.argmax(predict, axis = 1)[0])
                    
                    
                    
    if __name__ == '__main__':
        main()
        
    



    '''
    Unused code
    
    
            #Create T/F List for outliers using x_list (past y-values)
        #outliers = np.array(yBuild.find_outliers(idealMax, x_list))
    
            #yData = np.array(yBuild.classify())
    
            
        
        #Create Consecutive greens
        #consgreen2 = np.array(oldSet.find_consecutive_green(2))
        #consgreen4 = np.array(oldSet.find_consecutive_green(4))
        #consgreen6 = np.array(oldSet.find_consecutive_green(6))
        
        #Create consecutive reds
        #consred2 = np.array(oldSet.find_consecutive_red(2))
        #consred4 = np.array(oldSet.find_consecutive_red(4))
        #consred6 = np.array(oldSet.find_consecutive_red(6))
        
        #Create tierlist
        #tierList = np.array(oldSet.build_tiers(oldSet.dataSet, 1.2, 1.5, 1.9, 2.5, 5))
            #xData = yBuild.list_of_lists(consgreen2)
        #xData = yBuild.extend_list(consgreen4, xData)
        #xData = yBuild.extend_list(consgreen6, xData)
        #xData = yBuild.extend_list(consred2, xData)
        #xData = yBuild.extend_list(consred4, xData)
        #xData = yBuild.extend_list(consred6, xData)
        #xData = yBuild.extend_list(tierList, xData)
        #xData = yBuild.extend_list(outliers, xData)
    
    
    #This was used to run the algorithm in a continuous loop and determine the optimal level to cash out
    every game based on previous game results over num_games games.  Not profitable, but provided good insight
    
        if(train == 'Y'):
            game_hash = '9914966bf0f7f80fb9bca2180625b1731969cb584c80dd0d133a315996eb3262'
            oldSet = Helper_Methods.Helper_Methods()
            oldSet.build(game_hash, salt, num_variables)
            maxV = oldSet.find_chance_consecutive()           
            print("-"*80)
            totalCons = 0
            for a in maxV:
                totalCons += a[1]
                
            for a in maxV:
                print(f"It will reach {a[0]}: ",int((1000 * (a[1]/totalCons))), " times over 1000 games")
                print()
            
            print("-"*80)
    '''