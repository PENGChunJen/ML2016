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
    a = [['AMB_TEMP',   14,   14, ...]
         [     'CH4',  1.8,  1.8, ...]
         [      'CO', 0.51, 0.41, ...]
         ...
         [   'WS_HR',  0.5,  0.9, ...]]
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

def write_to_file(predict, filename):
    with open(filename, "wb") as f:
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

def cross_validation(X, Y):
    batch_size = 1000 
    validation_data = X[:batch_size,:] 
    validation_value = Y[:batch_size]
    
    train_data = X[batch_size:, :]
    train_value = Y[batch_size:] 
    return train_data, train_value, validation_data, validation_value

def run_gradient_descent(train_data, train_value, validation_data, validation_value, theta, learning_rate):
    num_points, num_features = train_data.shape
    hypothesis = train_data.dot(theta)
    loss = hypothesis - train_value 
    gradient = np.dot(train_data.T, loss)/num_points
    theta = theta - learning_rate * gradient
    train_cost = loss.T.dot(loss) / (2.*num_points)

    num_points, num_features = validation_data.shape
    validation_loss = validation_data.dot(theta) - validation_value
    validation_cost = validation_loss.T.dot(validation_loss) / (2.*num_points)
    return theta, train_cost, validation_cost

def gradient_descent(X, Y, theta, learning_rate, iterations):
    num_points, num_features = X.shape
    delta = 0.000001
    last_cost = float('Inf')
    train_data, train_value, validation_data, validation_value = cross_validation(X, Y)
    for it in xrange(iterations):
        theta, train_cost, validation_cost = run_gradient_descent(train_data, train_value, validation_data, validation_value, theta, learning_rate)
        
        #if abs(last_cost - cost) < delta:
        #    print 'last_cost:', last_cost, 'cost:', cost
        #    print("Converge at Iteration %5d | Train Cost: %f | Validation Cost: %f " %(it, train_cost, validation_cost))
        #    break
        if it%1000 == 0 :
            print("Iteration %5d | Train Cost: %f | Validation Cost: %f " %(it, train_cost, validation_cost))
            #print theta
            if validation_cost > last_cost:
                print "Rebound"
                print("Iteration %5d | Train Cost: %f | Validation Cost: %f " %(it, train_cost, validation_cost))
                break

        last_cost = validation_cost
    
    return theta


def gradient_descent_v2(X, Y, theta, learning_rate, iterations):
    num_points, num_features = X.shape
    delta = 0.000001
    last_cost = float('Inf')
    for it in xrange(iterations):
        hypothesis = X.dot(theta)
        loss = hypothesis - Y 
        gradient = np.dot(X.T, loss)/num_points
        theta = theta - learning_rate * gradient
        
        cost = loss.T.dot(loss) / (2.*num_points)
        #cost = np.sum(loss ** 2) / (2*num_points)
        if abs(last_cost - cost) < delta:
            print 'last_cost:', last_cost, 'cost:', cost
            print("Converge at Iteration %5d | Cost: %f" %(it, cost))
            break
        if it%1000 == 0 :
            print("Iteration %5d | Cost: %f" %(it, cost))
        last_cost = cost
    
    return theta

def gradient_descent_v1(X, Y, theta, learning_rate, iterations):
    num_points, num_features = X.shape
    for i in xrange(iterations):
        hypothesis = X.dot(theta)
        loss = hypothesis - Y
        gradient = np.dot(X.T, loss)/num_points
        theta = theta - learning_rate * gradient
        cost = loss.T.dot(loss) / (2.*num_points)
        #cost = np.sum(loss ** 2) / (2*num_points)
        if i%1000 == 0 :
            print("Iteration %5d | Cost: %f" %(i, cost))
    
    return theta

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
        
        if i%100 == 0 :
            print("Iteration %5d | Cost: %f" %(i, error))
    
    return w, b

def run():
    iterations = 100001
    learning_rate = 0.01
    X, Y = readTrainingData()
    print X.shape, Y.shape
    #w, b = gradient_descent_v0(X, Y, learning_rate, iterations)

    num_points, num_features = X.shape
    X, mean_r, std_r = normalize_feature(X)
    #print mean_r, std_r
    bias = np.array([np.ones(num_points)]).T
    X = np.concatenate((X, bias), axis=1)
    theta = np.zeros(num_features+1)
    #theta = gradient_descent_v1(X, Y, theta, learning_rate, iterations)
    theta = gradient_descent(X, Y, theta, learning_rate, iterations)
    
    testing_data = readTestingData()
    #print testing_data.shape
    num_points, num_features = testing_data.shape
    normalized_testing_data = (testing_data - mean_r)/std_r

    bias = np.array([np.ones(num_points)]).T
    X_testing = np.concatenate((normalized_testing_data, bias), axis=1)
    predict = (X_testing).dot(theta)
    write_to_file(predict, "linear_regression.csv")

if __name__ == '__main__':
    run()
