#--*- coding:utf-8 --*-

from __future__ import division
import numpy as np
from math import *

'''
A Simple Demo AdaBoost Algorithm Implementation
If time allowed,extend it to more general cases
This can only apply to the demo in the textbook
'''

def sign(x):
    if(x>=0):
        return 1
    else:
        return -1
    
    
class Classifier:
    def __init__(self,dataset,weight):
        self.dataset = dataset
        self.cweight = 0
        self.weight = weight
        self.threshold = 0
        self.errorrate = 0
        self.inverse = False
        
    def train(self):
        a = np.arange(1.5,9,1)
        error = 0
        for test in a:
            error = 0
            for i in range(10):
                if((self.dataset[0][i] < test and self.dataset[1][i] == -1) or (self.dataset[0][i] >test and self.dataset[1][i] == 1)):
                    error += self.weight[i]
                    
            if(self.errorrate == 0):
                self.errorrate = error
                self.threshold = test
            elif(self.errorrate > error):
                self.errorrate = error
                self.threshold = test
        
        another_error = 0
        for another_test in a:
            another_error = 0
            for i in range(10):
                if((self.dataset[0][i] < another_test and self.dataset[1][i] == 1) or (self.dataset[0][i] >another_test and self.dataset[1][i] == -1)):
                    another_error += self.weight[i]
                    
            if(self.errorrate > another_error):
                self.errorrate = another_error
                self.threshold = another_test
                self.inverse = True
        #For test use
        print("Threshold of this classfier is:",self.threshold)
        print("ErrorRate = ",self.errorrate)
                
    def getCWeight(self):
        self.cweight = 1/2*((log(1-self.errorrate))-(log(self.errorrate)))
        print("-----------------CWeight---------------------")
        print(self.cweight)
        
    def getUpdatedWeight(self):
        self.updateWeight()
        temp = [x/sum(self.weight) for x in self.weight]
        #For test case
        print(temp)
        return temp
        
    def updateWeight(self):
        for i in range(len(self.weight)):
            if self.dataset[0][i] < self.threshold:
                tmp = 1
            else:
                tmp = -1
            self.weight[i] = self.weight[i] * exp((-1)*self.cweight*self.dataset[1][i] * tmp)
        
    def predict(self,instance):
        if(not self.inverse):
            if(instance < self.threshold):
                return 1
            else:
                return -1
        else:
            if(instance < self.threshold):
                return -1
            else:
                return 1            
            
        
class StrongClassfier():
    def __init__(self,dataset,num = 3):
        self.num = num
        self.classfiers = []
        self.dataset = dataset
        self.weight = np.ones(10)/10
        
    def build_strong_classfier(self):
        for i in range(self.num):
            classfier = Classifier(self.dataset,self.weight)
            if(i==2):
                print("Hello")
            classfier.train()
            classfier.getCWeight()
            self.weight = classfier.getUpdatedWeight()
            self.classfiers.append(classfier)
            
    def predict_by_strong_classfier(self,instance):
        result = 0
        for sample in self.classfiers:
            result += sample.cweight * sample.predict(instance)
        return sign(result)
    
if __name__ == "__main__":
    dataset = np.array([[0,1,2,3,4,5,6,7,8,9],[1,1,1,-1,-1,-1,1,1,1,-1]])
    strongClassfier = StrongClassfier(dataset)
    strongClassfier.build_strong_classfier()
    #For Test Use
    test = range(0,10)
    tmp = [strongClassfier.predict_by_strong_classfier(x) for x in test]
    print(tmp)
        
        
        
                    
                    
                    
                   
                
                