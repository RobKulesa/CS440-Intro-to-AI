from NaiveBayes import NaiveBayesClassifier
import pandas as pd
import numpy as np
from collections import Counter
import time


def load_files(f):
    return [x[:-1]for x in open(f,'r').readlines()]
#returns df with each column corresponding to pixel in flattened image (feature), final column is the label

def create_dataframe(x,y,height):
    lst = []
    pixel_lst = []
    count = 0
    
    for i in x:
        for j in i:
            if j == ' ':
                pixel_lst.append(0)
            #elif j=='#':
            #    s+='2'
            else:
                pixel_lst.append(1)
        count+=1
        
        if count == height:
            lst.append(pixel_lst)
            count = 0
            pixel_lst = []
    
    df = pd.DataFrame(lst)
    df['label'] = y
    return df

def load_dataset(dataset,train_X_path,train_y_path,test_X_path,test_y_path):
    train_X,train_y,test_X,test_y = load_files(train_X_path),load_files(train_y_path),load_files(test_X_path),load_files(test_y_path)
    if dataset=='face':
        train_df = create_dataframe(train_X,train_y,70)
        test_df = create_dataframe(test_X,test_y,70)
    else:
        train_df = create_dataframe(train_X,train_y,28)
        test_df = create_dataframe(test_X,test_y,28)
        
    return train_df, test_df

def run_diagnostic(train_df,test_df):
    model = NaiveBayesClassifier()
    #train_range = [i for i in range(0,101,10)]
    test_X = test_df.iloc[:,:-1]
    test_y = test_df.iloc[:,-1]
    
    accs = []
    
    for i in range(10,101,10):
        acc = []
        for j in range(2):
            
            sampled_df = train_df.sample(frac=i/100)
            
            model.fit_df(sampled_df)
            preds = model.predict(test_X)
            acc.append(model.accuracy(preds,test_y))
        accs.append(acc)
            
    #print(accs)
    means = np.mean(accs,axis=1)
    stds = np.std(accs,axis=1)
    print('\t\t\t\t\t\t\tMEAN_ACC\tSTDEV')
    for i in range(len(means)):
        percent = 10*(i+1)
        
        print("Prediction distributions using %d%% training data:\t%s\t\t%s" % (percent,round(means[i],3),round(stds[i],3)))
    
                                               

def main():
    DIGIT_TRAIN_X_PATH = '../data/digitdata/trainingimages'
    DIGIT_TRAIN_Y_PATH = '../data/digitdata/traininglabels'
    DIGIT_TEST_X_PATH = '../data/digitdata/testimages'
    DIGIT_TEST_Y_PATH = '../data/digitdata/testlabels'
    
    FACE_TRAIN_X_PATH = '../data/facedata/facedatatrain'
    FACE_TRAIN_Y_PATH = '../data/facedata/facedatatrainlabels'
    FACE_TEST_X_PATH = '../data/facedata/facedatatest'
    FACE_TEST_Y_PATH = '../data/facedata/facedatatestlabels'
    
    
    
    dataset = input('Enter which dataset to run predictions on (face,digit): ')
    if dataset == 'face':
        train_df,test_df = load_dataset(dataset,FACE_TRAIN_X_PATH,FACE_TRAIN_Y_PATH,FACE_TEST_X_PATH,FACE_TEST_Y_PATH)
    elif dataset =='digit':
        train_df,test_df = load_dataset(dataset,DIGIT_TRAIN_X_PATH,DIGIT_TRAIN_Y_PATH,DIGIT_TEST_X_PATH,DIGIT_TEST_Y_PATH)
    else:
        print('That database is not available, please enter either ''face'' or ''digit'': ')
    
    
    diag = input('Visualize prediction distributions using different %s of training data? (y/n): ')
    if diag=='y':
        run_diagnostic(train_df,test_df)
    
    
    
    train_percent = input('Enter what percent of training data you want to use (10,20,30,etc): ')
    if train_percent=='':
        percent = 100
    else:
        percent = int(train_percent)
    train_df = train_df.sample(frac = (int(train_percent)/100))
    
    smoothing = input('Enter smoothing value (default is 1): ')
    if smoothing == '':
        smoothing = 1
        
    model = NaiveBayesClassifier()
    model.smooth = int(smoothing)
    model.fit_df(train_df)
    preds = model.predict(test_df.iloc[:,:-1],True)
    model.accuracy(preds,test_df.iloc[:,-1],True)
if __name__ == '__main__':
    main()
