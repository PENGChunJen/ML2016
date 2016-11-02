import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from lightning.regression import AdaGradRegressor
#from lightning.regression import SGDRegressor 

def readTrainingData():
    X = []
    Y = []
    with open("data/train.csv", "rb") as f:
        data = list(csv.reader(f))
    data = data[1:]
    a = np.array([np.array(data)[0:18,2]]).T
    for i in xrange(0,4320,18):
        d = np.array(data[i:i+18])[:,3:]
        d[d=='NR']=0.0
        a = np.concatenate((a,d),axis=1)
    '''
    a = [['AMB_TEMP', 14, 14, ....]
         ['CH4', 1.8, 1.8, ...]
         ['CO', 0.51, 0.41, ...]
         ...
         ['WS_HR', 0.5, 0.9, ...]]
    '''
    for i in xrange(1,5752,1):
        X.append(a[:,i:i+9].reshape(162))
        Y.append(a[9,i+9])
    
    return np.array(X).astype(np.float), np.array(Y).astype(np.float)

def readTestingData():
    testing_data = []
    with open("data/test_X.csv", "rb") as f:
        data = list(csv.reader(f))
    d = np.array(data)
    for i in xrange(0,240):
        a = d[i*18:(i+1)*18] 
        a[a=='NR']=0.0
        testing_data.append(a[:,2:].reshape(162))
    return np.array(testing_data).astype(np.float)

def write_to_file(predict):
    with open("result.csv", "wb") as f:
        f.write('id,value\n')
        for i in xrange(predict.shape[0]):
            f.write('id_%d,%f\n'%(i,predict[i]))

def run():
    iterations = 10001
    learning_rate = 0.01
    X_train, Y_train = readTrainingData()
    X_test = readTestingData()
    scaler = StandardScaler()
    scaler.fit_transform(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf = SGDRegressor(n_iter=100)
    
    #clf = AdaGradRegressor(n_iter=100)
    
    clf.fit(X_train, Y_train)
    print clf.score(X_train, Y_train)

    predict = clf.predict(X_test)
    write_to_file(predict)

if __name__ == '__main__':
    run()
