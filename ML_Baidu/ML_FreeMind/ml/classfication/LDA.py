#!/usr/bin/env python
#coding=utf-8
import numpy as np
from scipy import linalg

'''
@author: FreeMind
@version: 1.0
Created on 2014/3/9
A toy implementation of Linear Discriminant Analysis(LDA) algorithm
'''

class LDA:
    def __init__(self,test_data,targetDim):
        self.testData = test_data
        #Retrieve the class number of the test_data
        self.classSet = set([sample[-1] for sample in test_data])
        self.classNum = len(self.classSet)
        #Get the size of the training set
        self.sampleSize = len(test_data)
        #Get the dimension of the data
        self.dim = len(test_data[1])-1
        #Set the target dimension
        self.targetDim = targetDim
    
    #Train for the LDA algorithm
    def train(self):
        #Separate the data according to the classSet
        dict_sample = {}   
        dict_count = {}
        meanList = {}  
        totalMean = np.zeros(self.dim)
        for sample in self.testData:
            if sample[-1] not in dict_sample.keys():
                dict_sample[sample[-1]] = [sample[:-1]]
                dict_count[sample[-1]] = 1
            else:
                dict_sample[sample[-1]].append(sample[:-1])
                dict_count[sample[-1]] +=1
            if sample[-1] not in meanList.keys():
                meanList[sample[-1]] = []
                for i in range(self.dim):
                    meanList[sample[-1]].append(sample[i])
            else:
                for i in range(self.dim):
                    meanList[sample[-1]][i] += sample[i]
        for key in dict_count.keys():
            for i in range(self.dim):
                totalMean[i] += meanList[key][i]
                meanList[key][i] /= dict_count[key]           
        for i in range(self.dim):
            totalMean[i] /= self.sampleSize
        #Calculate the scatter matrix
        Sb = np.zeros([self.dim,self.dim])
        Sw = np.zeros([self.dim,self.dim])
        for key in dict_count.keys():
            tempArray = np.array(meanList[key])-np.array(totalMean)
            tempArray.shape = (self.dim,1)
            Sb += dict_count[key]*np.dot(tempArray,np.transpose(tempArray))
        for sample in self.testData:
            tempArray = np.array(meanList[sample[-1]])-np.array(sample[:-1])
            tempArray.shape=(self.dim,1)
            Sw+=np.dot(tempArray,np.transpose(tempArray))
        matrixFinal = np.dot(linalg.inv(Sw),Sb)
        eigenValue,eigenVector = linalg.eig(matrixFinal)
        print(eigenValue)
        print(eigenVector)
              
    #predict using the LDA algorithm
    def predict(self,predict_data):
        pass
        


#Test case
if __name__ == "__main__":
    test_data =[[2.95,6.63,"Qualified"],[2.53,7.79,"Qualified"],[3.57,5.65,"Qualified"],
            [3.16,5.47,"Qualified"],[2.58,4.46,"Unqualified"],
            [2.16,6.22,"Unqualified"],[3.27,3.52,"Unqualified"]]
    lda = LDA(test_data,1)
    lda.train()
    #lda.predict([[2.81,5.46]])
    
    