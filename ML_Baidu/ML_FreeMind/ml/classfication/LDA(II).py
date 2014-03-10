import numpy as np

class LDA:
    def runLDA(self,data,index):
        num_class = len(index)
        label = data[:,-1]
        data = data[:,:-1]
        uba = np.mean(data,axis = 0)
        k = data.shape[1]
        u = np.zeros([num_class,k])
        n = np.zeros(num_class)
        for i in index:
            u[i] = np.mean(data[label==i,:],axis=0)
            n[i] = sum(label==i)
        Sb = np.zeros([k,k])
        Sw = np.zeros([k,k])
        for i in range(num_class):
            Sb = Sb + n[i]*np.outer(u[i]-uba,u[i]-uba)

        for i in index:
            datai = data[label==i,:]
            for x in datai:
                Sw = Sw + np.outer(u[i]-x,u[i]-x)
        SWB = np.dot(np.linalg.inv(Sw),Sb)
        return np.linalg.eig(SWB), Sb, Sw


if __name__ == "__main__":
    data = np.asarray([[2.95,6.63,0],
                       [2.53,7.79,0],
                       [3.57,5.65,0],
                       [3.16,5.47,0],
                       [2.58,4.46,1],
                       [2.16,6.22,1],
                       [3.27,3.52,1]])
    index = np.asarray([0,1])
    lda = LDA()
    print(lda.runLDA(data,index))
