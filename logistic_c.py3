import numpy as np;
import pandas as pd;
import sys;
from scipy import stats;
import sklearn;
import csv;
import timeit;
import matplotlib.pyplot as plt;
from numpy import *;

#def encode(X):
#   res = zeros((X.shape[0],28));
#   for i in range(X.shape[1]):
#       if()
def f(theta,Y,X):
    return (1/(2*X.shape[0]))*np.sum(np.multiply(Y,ma.log(h(theta,X)).filled(0)));

def h(theta,X):
    temp = np.exp(X@theta.T);
    sum1 = np.sum(temp,axis = 1).reshape(temp.shape[0],1);
    temp = temp/sum1;
    return temp;

def g(theta,Y,X):
    return np.array(Y-h(theta,X)).T@X;

def hm(i,j):
    return 2/((1/i)+(1/j));

def logistic_c(train,test,op,wt):
    train = pd.read_csv(train,header = None);
    test = pd.read_csv(test,header = None);
    s = pd.read_csv('sample_pred.txt',header = None);

    X = train.values.copy();
    Y = X[:,X.shape[1]-1].copy();
    Xt = test.values.copy();
    A = {}
    for i in range(X.shape[1]):
        A[i] = {};

    A[0]['usual'] = 0;
    A[0]['pretentious'] = 1;
    A[0]['great_pret'] = 2;
    A[1]['proper'] = 3;
    A[1]['less_proper'] = 4;
    A[1]['improper'] = 5;
    A[1]['critical'] = 6;
    A[1]['very_crit'] = 7;
    A[2]['complete'] = 8;
    A[2]['completed'] = 9;
    A[2]['incomplete'] = 10;
    A[2]['foster'] = 11;
    A[3]['1'] = 12;
    A[3]['2'] = 13;
    A[3]['3'] = 14;
    A[3]['more'] = 15;
    A[4]['convenient'] = 16;
    A[4]['less_conv'] = 17;
    A[4]['critical'] = 18;
    A[5]['convenient'] = 19;
    A[5]['inconv'] = 20;
    A[6]['nonprob'] = 21;
    A[6]['slightly_prob'] = 22;
    A[6]['problematic'] = 23;
    A[7]['recommended'] = 24;
    A[7]['priority'] = 25;
    A[7]['not_recom'] = 26;
    A[8]['not_recom'] = 0;
    A[8]['recommend'] = 1;
    A[8]['very_recom'] = 2;
    A[8]['priority'] = 3;
    A[8]['spec_prior'] = 4;

    X_train = np.zeros((X.shape[0],27));
    Y_train = np.zeros((X.shape[0],5));
    X_test = np.zeros((Xt.shape[0],27));

    for i in range(X_train.shape[0]):
        for j in range(8):
            X_train[i,A[j][X[i,j]]] = 1;
        Y_train[i,A[8][X[i,8]]] = 1;
    
    for i in range(X_test.shape[0]):
        for j in range(8):
            X_test[i,A[j][Xt[i,j]]] = 1;
    
    X_train = np.insert(X_train,0,1,axis=1);
    X_test = np.insert(X_test,0,1,axis=1);

    n = X_train.shape[1];
    m = X_train.shape[0];
    k = Y_train.shape[1];
#    print(X_train.shape);
#    print(Y_train.shape);
#    print(theta.shape);

    mode = 2;
    alphas = [0.55,0.55,0.6];
    iterations = 10000;
    sizes = [175,100,200];

    # print(a);
    # print(b);    
    # print(iterations);
    for alpha in alphas:
        for size in sizes:
            theta = np.full((k,n),0.0);
            fs = [];
            print(alpha);
            print(size);
            sub_iters = m//size;   
            
            for i in range(int(iterations)):
                if(i%1000==0):
                    print(i/iterations);
                    #print(np.amax(X_train@theta.T));
                fs.append(f(theta,Y_train,X_train));
                for j in range(sub_iters):
                    x = X_train[j*size:(j+1)*size,:];
                    y = Y_train[j*size:(j+1)*size];
                    
                    if mode == 1:
            #            alpha = 0.0001;
                        theta = theta + (alpha/(size))*g(theta,y,x);

                    elif mode == 2:
                        theta = theta + (alpha/(np.sqrt(i+1)*size))*g(theta,y,x);

            plt.plot([i for i in range(int(iterations))],fs);
            plt.show();                
        #    print(X_test.shape);
        #    print(X_train.shape);
        #    print(theta.shape);
            temp = np.exp(X_test@theta.T);
            sum1 = np.sum(temp,axis=1).reshape(temp.shape[0],1);
            temp = temp/sum1;
            mat = np.argmax(temp,axis = 1);
            train_op = np.exp(X_train@theta.T);
            sum1 = np.sum(train_op,axis=1).reshape(train_op.shape[0],1);
            train_op = train_op/sum1;
            mat2 = np.argmax(train_op,axis = 1);

            output = [];    
            op2 = [];
            for i in range(mat.shape[0]):
                if mat[i] == 0:
                    output.append('not_recom');
                if mat[i] == 1:
                    output.append('recommend');
                if mat[i] == 2:
                    output.append('very_recom');
                if mat[i] == 3:
                    output.append('priority');
                if mat[i] == 4:
                    output.append('spec_prior');

            for i in range(mat2.shape[0]):
                if mat2[i] == 0:
                    op2.append('not_recom');
                if mat2[i] == 1:
                    op2.append('recommend');
                if mat2[i] == 2:
                    op2.append('very_recom');
                if mat2[i] == 3:
                    op2.append('priority');
                if mat2[i] == 4:
                    op2.append('spec_prior');

            np.savetxt(op,[p for p in output],delimiter=',',fmt='%s')
            np.savetxt(wt,[[i for i in line] for line in theta.T],delimiter = ',',fmt = '%s');

            t2 = open('sample_pred.txt');
            sample = [];
            for i in t2.read().split():
                sample.append(i);

            conf = np.zeros((5,5));
            B = {}
            B['not_recom'] = 0;
            B['recommend'] = 1;
            B['very_recom'] = 2;
            B['priority'] = 3;
            B['spec_prior'] = 4;
            for i in range(len(op2)):
                conf[B[Y[i]],B[op2[i]]] += 1;
            conf.astype(int);
            print(conf);
            # print(np.sum(conf)-np.sum(np.diagonal(conf)));

            f1 = [];
            recalls = [];
            precisions = [];
            tps = 0;
            fps = 0;
            fns = 0;
            for i in range(5):
                tp = conf[i,i];
                tps += tp;
                fp = np.sum(conf[:,i])-tp;
                fps += fp;
                fn = np.sum(conf[i,:])-tp;
                fns += fn;
                if tp+fn==0:
                    recall = 0;
                else:
                    recall = tp/(tp+fn);
                if tp+fp==0 :
                    precision = 0;
                else:
                    precision = tp/(tp+fp);
                if recall==0 or precision==0:
                    f1.append(0);
                else:
                    f1.append(hm(recall,precision));
                recalls.append(recall);
                precisions.append(precision);
            print(f1);
            if tps+fps!=0 and tps+fns!=0:
                print(hm(tps/(tps+fps),tps/(tps+fns)));
            else:
                print(0);
            if np.sum(recalls)!=0 and np.sum(precisions)!=0:
                print(hm(np.average(recalls),np.average(precisions)));
            else:
                print(0);

if __name__ == '__main__':
    logistic_c(*sys.argv[1:]);