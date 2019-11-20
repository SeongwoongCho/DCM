import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

class DCMoptimizer():
    def __init__(self, n_classes, weight, predict, true, metric):
        """
        weight : list or tuple or ndarray, normalized class weight of training set
        
        predict : ndarray with shape (M,N), predict result(result right after softmax) of valid set
        
        true : ndarray with shape (M,1), true label of valid set
        
        metric : evaluation function 
        
        
        """
        self.n_classes = n_classes
        self.weight = weight
        self.predict = predict
        self.true = true
        self.metric = metric
        
        self._get_confusion(predict,true)
        
    def _get_confusion(self,predict,true):
        predict = np.argmax(predict, axis=1)
        cond_mat = confusion_matrix(predict,true)
        self.cond_mat = cond_mat/np.sum(cond_mat,axis=1)[:,np.newaxis]
        
    def S(self,p,i,j,l):
        if i==j:
            return p[j]
        else:
            return l*(1-p[j])
    
    def search(self,space):
        best = -10000
        self.best_l = space[0]
        for l in space:
            print("searching on l = {}".format(l))
            self.l = l
            pred = self.apply(self.predict)
            score = self.metric(pred, self.true)
            if score > best:
                best = score
                self.best_l = self.l
        self.l = self.best_l
        
        print("search completed!")
        print("Final l = {} best score is {}".format(self.best_l,best))
        
    def apply(self, pred):
        """
        pred : ndarray with shape(M',N), predict result(result right after softmax) of test set
        
        return answer label applying this method 
        """
        final_pred = []
        for p in pred:
            label = []
            for i in range(self.n_classes):
                tmp = 0
                for j in range(self.n_classes):
                    tmp += self.S(p,i,j,self.l)*self.cond_mat[i,j]*self.weight[j]
                label.append(tmp)
            label = np.argmax(label)
            final_pred.append(label)
            
        return np.array(final_pred, dtype = np.int)