import sys, csv
import numpy as np
import numpy.ma as ma
import cPickle as pickle

def normalize(X):
    mean_r = [] 
    std_r = []
    X_norm = X
    num_features = X.shape[1]
    for i in xrange(num_features):
        m = np.mean(X[:,i])
        s = np.std(X[:,i])
        if s == 0:
            s = m
            m = 0
        mean_r.append(m)
        std_r.append(s)
        X_norm[:,i] = (X_norm[:,i]-m)/s
    return X_norm, np.array([mean_r]), np.array([std_r])

def sigmoid(z):
    # Numerically-stable sigmoid function
    return np.exp(-np.logaddexp(0,-z))
    #return 1 / (1 + np.exp(-z))

def cross_entropy(x, y):
    # y is binary classification
    return -( y*(ma.log(x).filled(0)) + (1-y)*(ma.log(1-x).filled(0)))

def compute_cost(theta, X, Y):
    num_points, num_features = X.shape
    Z = sigmoid(X.dot(theta))
    loss = cross_entropy(Z, Y)
    return sum(loss)/num_points

def logistic_regression(theta, learning_rate, X, Y):
    num_points, num_features = X.shape
    Z = sigmoid(X.dot(theta))
    loss = cross_entropy(Z, Y)
    cost = sum(loss)/num_points
    #gradient = np.dot((Z - Y),X)/num_points
    gradient = np.dot((Z - Y),X)
    theta = theta - learning_rate * gradient
    ''' 
    #np.set_printoptions(threshold='nan')
    print 'X',X.shape,'\n', X
    print 'Y',Y.shape,'\n', Y
    print 'Z',Z.shape, '\n',Z
    print 'theta', theta.shape, '\n', theta 
    print 'gradient', gradient.shape, '\n', gradient 
    print 'loss', loss.shape, '\n', loss 
    print 'cost', train_cost
    '''
    return theta, cost
   

def train_model(training_data):
    np.random.shuffle(training_data) 
    num_points = training_data.shape[0]
    num_features = 58
    X = training_data[:,1:num_features]
    X_labels = training_data[:,num_features]
    
    bias = np.array([np.ones(num_points)]).T
    X = np.concatenate((X, bias), axis=1)
    
    fold = 10
    train_data = X[num_points/fold:,:]
    train_labels = X_labels[num_points/fold:]
    val_data = X[:num_points/fold,:]
    val_labels = X_labels[:num_points/fold]
    
    num = val_data.shape[0]
    max_iterations = 1000000 
    DELTA = 1e-8
    learning_rate = 0.0005
    print("Total Num:%10d | Train Data: %12d | Validation Data: %12d | alpha: %.10f" %(X.shape[0], train_data.shape[0], val_data.shape[0], learning_rate))

    theta = np.zeros(num_features)
    #theta = np.random.rand(num_features)
    train_data, mean, std = normalize(train_data)
    val_data = (val_data - mean)/std

    last_val_cost = float('INF')
    last_train_cost = float('INF')
    last_it = 0
    train_cost_hist = []
    val_cost_hist = []

    for it in xrange(max_iterations):
        theta, train_cost = logistic_regression(theta, learning_rate, train_data, train_labels)
        val_cost = compute_cost(theta, val_data, val_labels)
        
        
        if it%1000 == 0 :
            model = mean, std, theta
            threshold, diff = sweepThreshold(model)
            
            if abs(last_train_cost-train_cost) < DELTA: #converge
                print("Iteration %10d | Train Cost: %.10f | Validation Cost: %.10f | thres: %.3f | diff: %4d " %(it, train_cost, val_cost, threshold, diff))
                break
            '''
            #elif abs(last_train_cost - train_cost)*num < learning_rate*learning_rate or ((it - last_it)>10000 and last_val_cost < val_cost):
            elif (it - last_it)>2000 and last_val_cost < val_cost:
                learning_rate = learning_rate/10
                #val_cost = last_val_cost
                last_it = it
                print("Iteration %10d | Train Cost: %.10f | Validation Cost: %.10f | thres: %.3f | diff: %4d | alpha: %.10f" %(it, train_cost, val_cost, threshold, diff, learning_rate))
            '''
            print("Iteration %10d | Train Cost: %.10f | Validation Cost: %.10f | thres: %.3f | diff: %4d " %(it, train_cost, val_cost, threshold, diff))
            
            train_cost_hist.append((it, train_cost))
            val_cost_hist.append((it, val_cost))
            last_train_cost = train_cost
            last_val_cost = val_cost

    #with open('train_cost_history', 'wb') as f:
    #    pickle.dump( train_cost_hist, f )
    #with open('val_cost_history', 'wb') as f:
    #    pickle.dump( val_cost_hist, f )
    
    model = mean, std, theta
    return model

def predict(model, testing_data, threshold = 0.0):
    mean, std, theta = model
    testing_data = testing_data[:,1:]
    num_points, num_features = testing_data.shape
    
    bias = np.array([np.ones(num_points)]).T
    X = np.concatenate((testing_data, bias), axis=1)
    X = (X - mean)/std
    
    labels = sigmoid(X.dot(theta))
    #with open('raw_labels.csv', 'wb') as f:
    #    f.write('id,label\n')
    #    for i in xrange(labels.shape[0]):
    #        f.write('%d,%f\n'%(i+1,labels[i]))

    #labels = np.around(labels+threshold)
    #print 'labels',labels.shape, '\n',labels
    return labels

def train(argv):
    training_file = argv[1]
    output_model = argv[2]
    
    with open(training_file, 'rb') as f:
        training_data = np.array(list(csv.reader(f))).astype(float)
    
    model = train_model( training_data ) 

    with open(output_model, 'wb') as f:
        pickle.dump( model, f )

def sweepThreshold(model):
    mean, std, theta = model
    training_file = 'spam_data/spam_train.csv'
    with open(training_file, 'rb') as f:
        training_data = np.array(list(csv.reader(f))).astype(float)
    num_features = 58
    X = training_data[:,:num_features]
    X_labels = training_data[:,num_features]
    raw_labels = predict(model, X)
    
    threshold_diff = []
    for i in xrange(500):
        threshold = i/1000.0
        labels = np.around(raw_labels+threshold)
        diff = np.logical_xor(X_labels, labels).astype(int)
        threshold_diff.append((threshold, sum(diff)))
        #print threshold, sum(diff)
    threshold, d = min(threshold_diff, key = lambda k: k[1])
    #print 'threshold:', threshold,'different training labels:', d, '/', X_labels.size
    '''
    labels = np.around(raw_labels+threshold)
    diff = np.logical_xor(X_labels, labels).astype(int)
    for i in xrange(len(diff)):
        if diff[i]:
            #print i, X_labels[i], labels[i], raw_labels[i]
    '''
    return threshold, d

def test(argv):
    model = pickle.load(open(argv[1], 'rb'))
    testing_file = argv[2]
    prediction = argv[3]
    
    with open(testing_file, 'rb') as f:
        testing_data = np.array(list(csv.reader(f))).astype(float)
    
    threshold, diff = sweepThreshold(model)
    print 'threshold:', threshold,'different training labels:', diff
    labels = predict(model, testing_data, threshold)
    
    labels = np.around(labels+threshold)
    with open(prediction, 'wb') as f:
        f.write('id,label\n')
        for i in xrange(labels.shape[0]):
            f.write('%d,%d\n'%(i+1,labels[i]))

if __name__ == '__main__':
    if len(sys.argv) == 3:
        train(sys.argv)
    elif len(sys.argv) == 4:
        test(sys.argv)
    else:
        print 'INPUT FORMAT ERROR!'
        print 'Usage 1: ./train.sh <training data> <output model>'
        print 'Usage 2: ./test.sh <model name> <testing data> prediction.csv'
