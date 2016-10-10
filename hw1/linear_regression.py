import numpy as np
import csv

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

def normalize_feature(X):
    mean_r = [] 
    std_r = []
    X_norm = X
    num_features = X.shape[1]
    for i in xrange(num_features):
        m = np.mean(X[:,i])
        s = np.std(X[:,i])
        mean_r.append(m)
        std_r.append(s)
        X_norm[:,i] = (X_norm[:,i]-m)/s
    return X_norm, np.array([mean_r]), np.array([std_r])

def compute_cost(X, Y, theta):
    m = Y.size
    error = X.dot(theta) - Y
    J = (1.0 /(2*m)) * errors.T.dot(errors)
    return J

def gradient_descent(X, Y, theta, learning_rate, iterations):
    num_points, num_features = X.shape
    for i in xrange(iterations):
        hypothesis = X.dot(theta)
        loss = hypothesis - Y
        cost = loss.T.dot(loss) / (2.*num_points)
        #cost = np.sum(loss ** 2) / (2*m)
        #print("Iteration %5d | Cost: %f" %(i, cost))
        if i%1000 == 0 :
            print("Iteration %5d | Cost: %f" %(i, cost))
        gradient = np.dot(X.T, loss)/num_points
        theta = theta - learning_rate * gradient
    return theta, cost

def gradient_descent_v0(X, Y, learning_rate, iterations):
    num_points, num_features = X.shape
    
    w = np.zeros(num_features)
    b = np.zeros(num_features)
    w_grad_acc = np.zeros(num_features)
    b_grad_acc = np.zeros(num_features)

    for i in xrange(iterations):
        w_grad = np.zeros(num_features)
        b_grad = np.zeros(num_features)
        
        for n in xrange(num_points):
            w_grad = w_grad - 2.0*(Y[n] - b - w*X[n,:]) * X[n,:]  
            b_grad = b_grad - 2.0*(Y[n] - b - w*X[n,:])
        
        w_grad_acc += w_grad**2
        b_grad_acc += b_grad**2
        
        b = b - learning_rate/np.sqrt(b_grad_acc) * b_grad
        w = w - learning_rate/np.sqrt(w_grad_acc) * w_grad
   
        error =  0
        for n in xrange(num_points):
            error += (Y[n] - (w*X[n,:] + b)) ** 2
        error = sum(error)/float(num_points)
        
        #if i%1000 == 0 :
        print("Iteration %5d | Cost: %f" %(i, error))
    
    return w, b

def run():
    iterations = 10001
    learning_rate = 0.01
    X, Y = readTrainingData()
    print X.shape, Y.shape
    #w, b = gradient_descent_v0(X, Y, learning_rate, iterations)

    num_points, num_features = X.shape
    X, mean_r, std_r = normalize_feature(X)
    bias = np.array([np.ones(num_points)]).T
    X = np.concatenate((X, bias), axis=1)
    theta = np.zeros(num_features+1)
    theta, cost = gradient_descent(X, Y, theta, learning_rate, iterations)
    
    testing_data = readTestingData()
    num_points, num_features = testing_data.shape
    normalized_testing_data = (testing_data - mean_r)/std_r
    bias = np.array([np.ones(num_points)]).T
    X_testing = np.concatenate((normalized_testing_data, bias), axis=1)
    predict = (X_testing).dot(theta)
    write_to_file(predict)

if __name__ == '__main__':
    run()
    a = np.array([[1,2],[3,3]])
    b = np.array([[1,1],[2,2]])
    print np.subtract(a,b)
