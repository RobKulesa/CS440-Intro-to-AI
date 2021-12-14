import pandas as pd
import numpy as np
from collections import Counter
import time


class NaiveBayesClassifier():    
    def fit_df(self,df):
        features = df.iloc[:, :-1]
        target = df.iloc[:, -1]
        self.fit(features,target)
        
    def fit(self,features,target):
        self.classes = sorted(np.unique(target))
        self.features = features
        self.feature_num = features.shape[1]
        self.sample_count = features.shape[0]
        self.df = features.assign(label=target)
        self.get_class_priors(self.df)
        self.smooth = 1
        self.create_likelihood_table()
    
    def predict(self, X,verbose=False):
        #passes every sample in feature vector into get_posterior
        start_time = time.time()

        preds =  [self.get_posterior(x) for x in X.to_numpy()]
        if verbose:
            print("Predictions took %s seconds for %d samples" % (time.time() - start_time,len(X)))
        return preds
    
    def accuracy(self,preds,y,verbose=False):
        count = 0
        for x in range(len(preds)):
            if preds[x]==y[x]:
                count+=1
        acc = count/len(y)
        
        if verbose:
            print('Testing accuracy: %f (%d/%d)' % (acc,count,len(y)))
        return acc
    
    def get_class_priors(self,df):
        self.class_priors = []
        for i in self.classes:
            self.class_priors.append( len(df[df['label']==i]) / len(df) )
        return self.class_priors
    
    def get_posterior(self, x):
        posteriors = []
        #for every class calculates likelihood for every pixel in a sample
        for class_idx in range(len(self.classes)):
            prior = np.log(self.class_priors[class_idx])
            
            
            #passes single sample and returns log likelihood of every pixel in sample
            likelihood = np.sum(np.log(self.access_likelihoods(class_idx,x)))
            
            posterior = prior + likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]    

    def create_likelihood_table(self):
        prob_table = dict.fromkeys(self.classes)
        off_pixels = np.asarray([0]*self.feature_num)
        on_pixels = np.asarray([1]*self.feature_num)
        
        self.class_likelihoods = []
        
        for class_idx in range(len(self.classes)):
            off_pixel_prob = self.get_likelihoods(class_idx,off_pixels)
            on_pixel_prob = self.get_likelihoods(class_idx,on_pixels)
            off_on_probs = list(zip(off_pixel_prob,on_pixel_prob))

            self.class_likelihoods.append(off_on_probs)
    
    def get_likelihoods(self,class_idx,x):
        likelihoods = []
        
        #subsets training set into only samples that match the current class
        train = self.df[self.df['label']==self.classes[class_idx]].iloc[:,:-1].to_numpy()
        
        
        #calculate likelihood of observed pixel in single sample based off train set
        for pixel_idx in range(len(x)): #0-4199
            likelihood = (list(train[:,pixel_idx]).count(x[pixel_idx])+self.smooth) / train.shape[0]
            likelihoods.append(likelihood)
            
        return likelihoods
    
    
    def access_likelihoods(self,class_idx,x):
        likelihoods = []
        
        for pixel_idx in range(len(x)):
            if x[pixel_idx] == 0:
                likelihood = self.class_likelihoods[class_idx][pixel_idx][0]
            else:
                likelihood = self.class_likelihoods[class_idx][pixel_idx][1]
            likelihoods.append(likelihood)
        return likelihoods
    


    #####################################################################
    #def old_predict(self, X):
    #    #passes every sample in feature vector into get_posterior
    #    preds =  [self.get_posterior(x) for x in tqdm(X.to_numpy())]
    #    return preds
   # 
   # def old_get_posterior(self, x):
   #     posteriors = []
   #     #for every class calculates likelihood for every pixel in a sample
   #     for class_idx in range(len(self.classes)):
   #         prior = np.log(self.class_priors[class_idx])
   #         
   #         
   #         #passes single sample and returns log likelihood of every pixel in sample
   #         likelihood = np.sum(np.log(self.get_likelihoods(class_idx,x)))
   #         posterior = prior + likelihood
   #         posteriors.append(posterior)
   #     return self.classes[np.argmax(posteriors)]
    
    #takes as input a single sample of feature length 4200

    
    #####################################################################
    #generate table of likelihoods for each class, with likelihood of each pixel for each class (i.e. dictionary of classes with each key containing list of 
    #tuples of likelihoods [0]=pixel off [1] = pixel on)
    

    
