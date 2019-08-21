import numpy as np;
import pandas as pd;
import sys;
import scipy;
import sklearn;
import csv;
import timeit;

def f(theta,Y,X):
    return np.power(np.linalg.norm(Y-h(theta,X)),2);

def h(theta,X):
    temp = np.exp(X@theta.T);
    sum1 = np.sum(temp,axis=1).reshape(temp.shape[0],1);
    temp = temp/sum1;
    return temp;

def g(theta,Y,X):
    return np.array(Y-h(theta,X)).T@X;

def logistic_a(train,test,op,wt):
    train = pd.read_csv(train,header = None);
    test = pd.read_csv(test,header = None);

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
    theta = np.full((k,n),0);
#    print(X_train.shape);
#    print(Y_train.shape);
#    print(theta.shape);

    mode = 2;
    alpha = 0.55;
    iterations = 10000;
    size = 175;
    sub_iters = m//size;

    # print(mode);
    # print(a);
    # print(b);
    # print(alpha);
    print(iterations);
    # print(size);

    for i in range(int(iterations)):
        # if(i%1000==0):
        #     print(i//1000);
        #     print(np.amax(X_train@theta.T));
        a=0.5;
        b=0.5;
        for j in range(sub_iters):
            x = X_train[j*size:(j+1)*size,:];
            y = Y_train[j*size:(j+1)*size];
            
            if mode == 1:
    #            alpha = 0.0001;
                theta = theta + (alpha/(size))*g(theta,y,x);

            elif mode == 2:
                theta = theta + (alpha/(np.sqrt(i+1)*size))*g(theta,y,x);

            elif mode == 3:
                t = a;
                while f(theta+a*g(theta,y,x),y,x)>f(theta,y,x)+a*b*f(theta,y,x)*np.power(np.linalg.norm(x),2):
                    a = a*t;   
                theta = theta + (a/size)*g(theta,y,x);
                    
#    print(X_test.shape);
#    print(X_train.shape);
#    print(theta.shape);
    temp = np.exp(X_test@theta.T);
    sum1 = np.sum(temp,axis=1).reshape(temp.shape[0],1);
    temp = temp/sum1;
    mat = np.argmax(temp,axis = 1);

    output = [];
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

    np.savetxt(op,[p for p in output],delimiter=',',fmt='%s')
    np.savetxt(wt,[[i for i in line] for line in theta.T],delimiter = ',',fmt = '%s');
    
    # t1 = open('output.csv', 'r');
    # t2 = open('sample_pred.txt', 'r');
    # fileone = t1.readlines();
    # filetwo = t2.readlines();
    # t1.close();
    # t2.close();

    # x = 0;
    # count = 0;
    # for i in fileone:
    #     if i != filetwo[count]:
    #         x=x+1;
    #     count=count+1;    
    # print(x);

if __name__ == '__main__':
    logistic_a(*sys.argv[1:]);