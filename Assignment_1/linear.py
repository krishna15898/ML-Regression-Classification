import pandas as pd;
import numpy as np;
import os;
from sklearn import linear_model;
import sys;

def featureExt(X):
	#deleting constant feature
	[r,c] = X.shape;
	for i in range(c):
		check = X[0,i];
		flag = True;
		for j in range(X.shape[0]-1): 
			if X[j+1,i] != X[j,i]:
				flag = False;
		if flag:
			np.delete(X,i,1);

	#adding powers of features
	for i in range(2,4):
		X = np.concatenate((X,X[:,:X.shape[1]]**i),1);

	return X;

def normal(trainData,testData,output,weight):
	train = pd.read_csv(trainData,header=None);
	X = train.values;
	[r,c] = X.shape;
	#print(c);
	Y = X[:,c-1].copy();
	X = X[:,:c-1];
	X = np.insert(X,0,1,axis=1);
	W = np.linalg.pinv( (X.T)@X ) @ (X.T) @ Y;
	
	test = pd.read_csv(testData,header=None);
	[rt,ct] = test.shape;
	X_test = test.values;
	X_test = np.insert(X_test,0,1,axis=1);
	
	for i in W:
		print(i,file=open(weight,"a"));
	prediction = X_test@W;
	for j in prediction:
		print(j,file=open(output,"a"));

def ridge(trainData,testData,reg,op,wt):

	def calculate_error(X,Y,W,lmbda):
		error1 = np.linalg.norm( X @ W - Y);
		return (error1*error1);

	train = pd.read_csv(trainData,header=None);
	X = train.values;
	[r,c] = X.shape;
	Y = X[:,c-1].copy();
	X = X[:,:c-1];
	X = np.insert(X,0,1,axis=1);

	reg_txt = open(reg);
	lmbdas = [];
	for word in reg_txt.read().split():
		lmbdas.append(float(word));
	min_error = float("inf");
	lmbda_best = lmbdas[0];
	
	length = r//10;
	for lmbda in lmbdas:
		error = 0; 
		
		for i in range(9):#only K-1 folds are validated here
			X_t = np.concatenate((X[0:i*length,:],X[(i+1)*length:,:]),axis=0);
			Y_t = np.concatenate((Y[0:i*length],Y[(i+1)*length:]),axis=0);
			W = np.linalg.pinv( (X_t.T) @ X_t + lmbda*
				np.eye(c)) @ (X_t.T) @ Y_t;
			error += calculate_error(X[i*length:(i+1)*length-1,:],Y[i*length:(i+1)*length-1],W,lmbda);
		
		X_t = X[0:9*length,:];
		Y_t = Y[0:9*length];
		W = np.linalg.pinv( (X_t.T) @ X_t + lmbda*np.eye(c)) @ (X_t.T) @ Y_t;
		
		error += calculate_error(X[9*length:r,:],Y[9*length:r],W,lmbda);
		error /= Y.T@Y;
		#print(error);
		#print(lmbda);
		if error <= min_error:
			lmbda_best = lmbda;
			min_error = error;
	
	print(lmbda_best);
	#print(min_error);

	W = np.linalg.pinv( X.T @ X + lmbda_best * np.eye(c) ) @ X.T @ Y;

	test = pd.read_csv(testData,header=None);
	[rt,ct] = test.shape;
	X_test = test.values;
	X_test = np.insert(X_test,0,1,axis=1);
	
	prediction = X_test@W;

	for w in W:
		print(w,file=open(wt,"a"));
	for p in prediction:
		print(p,file=open(op,"a"));

def lasso(trainData,testData,op):

	train = pd.read_csv(trainData,header=None);
	X = train.values;
	[r,c] = X.shape;
	Y = X[:,c-1].copy();
	X = X[:,:c-1];

	#print(X.shape);
	X = featureExt(X);	

	X = np.insert(X,0,1,axis=1);
	#print(X.shape);
	#parameter setting
	lmbdas = [1e-4,3e-4,1e-3,3e-3,0.01,0.03,0.1,0.3,1,3,10,30,100];
	min_error = float('inf');
	lmbda_best = lmbdas[0];
	length = X.shape[0]//10;
	#K fold validation
	for lmbda in lmbdas:
		error = 0; 
		
		for i in range(9):#only K-1 folds are validated here
			X_t = np.concatenate((X[0:i*length,:],X[(i+1)*length:,:]),axis=0);
			Y_t = np.concatenate((Y[0:i*length],Y[(i+1)*length:]),axis=0);
			reg = linear_model.LassoLars(alpha = lmbda,max_iter = 1000);
			reg.fit(X[i*length:(i+1)*length-1,:],Y[i*length:(i+1)*length-1]);
			pred = reg.predict(X[i*length:(i+1)*length-1,:]); 
			error += (Y[i*length:(i+1)*length-1]-pred).T@(Y[i*length:(i+1)*length-1]-pred);
		#Kth fold written now outside for loop
		X_t = X[0:9*length,:];
		Y_t = Y[0:9*length];
		reg = linear_model.LassoLars(alpha = lmbda,max_iter = 1000);
		reg.fit(X_t,Y_t);
		pred = reg.predict(X[9*length:,:]);
		error += (Y[9*length:]-pred).T@(Y[9*length:]-pred);
		error /= (Y.T@Y);
		if error <= min_error:
			lmbda_best = lmbda;
			min_error = error;
	
	print(lmbda_best);
	#print(min_error);
	reg_test = linear_model.LassoLars(alpha=lmbda_best);
	reg_test.fit(X,Y);

	test = pd.read_csv(testData,header=None);
	X_test = test.values;
	#print(X_test.shape);
	[rt,ct] = X_test.shape;
	X_test = featureExt(X_test);

	X_test = np.insert(X_test,0,1,axis=1);
	#print(X_test.shape);
	prediction = reg_test.predict(X_test); 

	for p in prediction:
		p = p * (p > 0)
		print(p,file=open(op,"a"));

	
if __name__ == '__main__':
    if sys.argv[1] == 'a':
        normal(*sys.argv[2:])
    elif sys.argv[1] == 'b':
        ridge(*sys.argv[2:])
    elif sys.argv[1] == 'c':
        lasso(*sys.argv[2:])


