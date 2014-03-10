import numpy as np
import math
import operator

class DTree_ID3:
    def runDT(self, dataset, features):
        classList = [sample[-1] for sample in dataset]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataset[0]) == 1:
            return self.classify(classList)

        max_index = self.Max_InfoGain(dataset) ##index
        max_fea = features[max_index]
        myTree = {max_fea:{}}
        fea_val = [sample[max_index] for sample in dataset]
        unique = set(fea_val);    
        del (features[max_index]) 
        for values in unique:          
            sub_dataset = self.splitDataSet(dataset, max_index, values)         
            myTree[max_fea][values] = self.runDT(sub_dataset,features) 
        features.insert(max_index,max_fea)  
        return myTree

    def classify(self,classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), revese = True)
        return sortedClassCount[0][0]

    def Max_InfoGain(self, data_set):
        #compute all features InfoGain, return the maximal one
        Num_Fea = len(data_set[1,:])
        #Num_Tup = len(data_set)
        max_IG = 1
        max_Fea = -1
        for i in range(Num_Fea-1):
            InfoGain = self.Info(data_set[:,[i,-1]])
            if (max_IG > InfoGain):
                max_IG = InfoGain
                max_Fea = i
        return max_Fea
    
    def Info(self, data):
        dic = {}
        for tup in data:
            if tup[0] not in dic.keys():
                dic[tup[0]] = {}
                dic[tup[0]][tup[1]] = 1
            elif tup[1] not in dic[tup[0]]:
                dic[tup[0]][tup[1]] = 1
            else:
                dic[tup[0]][tup[1]] += 1

        S_total = 0.0
        for key in dic:
            s = 0.0
            for label in dic[key]:
                s += dic[key][label]
            S_each = 0.0
            for label in dic[key]:
                prob = float(dic[key][label]/s)
                if prob !=0 :
                    S_each -= prob*math.log(prob,2)
            S_total += s/len(data[:,0])*S_each
        return S_total
    
    def splitDataSet(self,dataSet,featureIndex,value):
        subDataSet = []
        dataSet = dataSet.tolist()
        for sample in dataSet:
            if sample[featureIndex] == value:
                reducedSample = sample[:featureIndex]  
                reducedSample.extend(sample[featureIndex+1:])  
                subDataSet.append(reducedSample)
        return np.asarray(subDataSet)

if __name__ == "__main__":
    dataSet = np.array([["Cool","High","Yes","Yes"],["Ugly","High","No","No"],
               ["Cool","Low","No","No"],["Cool","Low","Yes","Yes"],
               ["Cool","Medium","No","Yes"],["Ugly","Medium","Yes","No"]])
    featureSet = ["Appearance","Salary","Office Guy"]
    dTree = DTree_ID3()
    tree = dTree.runDT(dataSet,featureSet)
    print(tree)