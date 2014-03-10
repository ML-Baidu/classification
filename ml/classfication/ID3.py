import numpy as np
import math

'''
@author: FreeMind
@version: 1.1
Created on 2014/2/26
This is a basic implementation of the ID3 algorithm.
'''

class DTree_ID3:
    def __init__(self,dataSet,featureSet):
        #Retrieve the class list
        self.classList = set([sample[-1] for sample in dataSet])
        if ((len(featureSet)+1) != (len(dataSet[1]))):
            print("The feature set do not match with the data set,please check your data!")
        #Set the feature set
        self.featureSet = featureSet

    def runDT(self,dataSet):
        ##recursive run DT, return a dictionary object
        ##1.check boundary: two case - only one tuple
        ##  or all tuples have the same label
        ##2.compute inforGain for each feature, select
        ##  feature corresponding to max inforGain
        ##3.split data set on that feature, recursively
        ##  call runDT on subset until reaching boundary   
        classList = set([sample[-1] for sample in dataSet])
        #If the data set is pure or no features
        if len(classList)==1:
            return list(classList)[-1]
        if len(self.featureSet)==0:  #no more features
            return self.voteForMost(dataSet,classList)
      
        bestSplit = self.findBestSplit(dataSet)
        bestFeatureLabel = self.featureSet[bestSplit]
        dTree = {bestFeatureLabel:{}}
        featureValueList = set([example[bestSplit] for example in dataSet])    
        for value in featureValueList: 
            subDataSet = self.splitDataSet(dataSet,bestSplit,value)
            self.featureSet.remove(self.featureSet[bestSplit])   
            dTree[bestFeatureLabel][value] = self.runDT(subDataSet)
            self.featureSet.insert(bestSplit,bestFeatureLabel)  
        return dTree 
    
    #Find the most in the set
    def voteForMost(self,dataSet,classList):
        count = np.zeros(len(self.classList))
        for sample in dataSet:
            for index in range(len(self.classList)):
                if sample[-1]==list(self.classList)[index]:
                    count[index] = count[index]+1
        maxCount = 0
        tag = 0
        for i in range(len(count)):
            if count[i]>maxCount:
                maxCount = count[i]
                tag = i
        return list(self.classList)[tag]
    
    #Compute all infoGains,return the best split   
    def findBestSplit(self,dataSet):
        #initialize the infoGain of the features
        infoGain = 0
        tag = 0
        for i in range(len(self.featureSet)):
            tmp = self.infoGain(dataSet,i)
            if(tmp > infoGain):
                infoGain = tmp
                tag = i
        return tag
            
    def entropy(self,dataSet):
        total = len(dataSet)
        cal = np.zeros(len(self.classList))
        entropy = 0
        for sample in dataSet:
            index = list(self.classList).index(sample[-1],)
            cal[index] = cal[index]+1
        for i in cal:
            proportion = i/total
            if proportion!=0:
                entropy -= proportion*math.log(proportion,2) 
        return entropy        

    def infoGain(self,dataSet,featureIndex):
        origin = self.entropy(dataSet)
        featureValueList = set([sample[featureIndex] for sample in dataSet])
        newEntropy = 0.0  
        for val in featureValueList:  
            subDataSet = self.splitDataSet(dataSet,featureIndex,val)  
            prob = len(subDataSet)/(len(dataSet))  
            newEntropy += prob*self.entropy(subDataSet)
        return origin-newEntropy
       
    def splitDataSet(self,dataSet,featureIndex,value):
            subDataSet = []
            for sample in dataSet:
                if sample[featureIndex] == value:
                    reducedSample = sample[:featureIndex]  
                    reducedSample.extend(sample[featureIndex+1:])  
                    subDataSet.append(reducedSample)
            return subDataSet

    def predict(self,testData):
        ##predict for the test data
        pass

if __name__ == "__main__":
    dataSet = [["Cool","High","Yes","Yes"],["Ugly","High","No","No"],
               ["Cool","Low","No","No"],["Cool","Low","Yes","Yes"],
               ["Cool","Medium","No","Yes"],["Ugly","Medium","Yes","No"]]
    featureSet = ["Appearance","Salary","Office Guy"]
    dTree = DTree_ID3(dataSet,featureSet)
    tree = dTree.runDT(dataSet)
    print(tree)