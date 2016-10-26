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
    gradient = np.dot((Z - Y),X)/num_points
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
    
    fold = 5
    train_data = X[:num_points/fold,:]
    train_labels = X_labels[:num_points/fold]
    val_data = X[num_points/fold:,:]
    val_labels = X_labels[num_points/fold:]
    
    max_iterations = 1000000000000
    learning_rate = 0.0001
    last_val_cost = float('INF')
    #theta = np.ones(num_features)
    theta = np.random.rand(num_features)
    train_data, mean, std = normalize(train_data)
    val_data = (val_data - mean)/std


    for it in xrange(max_iterations):
        theta, train_cost = logistic_regression(theta, learning_rate, train_data, train_labels)
        val_cost = compute_cost(theta, val_data, val_labels)
        if it%100 == 0 :
            if it > 10000 and last_val_cost < val_cost:
                print("Iteration %5d | Train Cost: %.10f | Validation Cost: %.10f" %(it, train_cost, val_cost))
                break
            last_val_cost = val_cost
            if it%1000 == 0:
                print("Iteration %5d | Train Cost: %.10f | Validation Cost: %.10f" %(it, train_cost, val_cost))

    model = mean, std, theta
    return model

def predict(model, testing_data):
    mean, std, theta = model
    testing_data = testing_data[:,1:]
    num_points, num_features = testing_data.shape
    
    bias = np.array([np.ones(num_points)]).T
    X = np.concatenate((testing_data, bias), axis=1)
    X = (X - mean)/std
    
    labels = sigmoid(X.dot(theta))
    with open('raw_labels.csv', 'wb') as f:
        f.write('id,label\n')
        for i in xrange(labels.shape[0]):
            f.write('%d,%f\n'%(i+1,labels[i]))

    labels = np.around(labels)
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

def test(argv):
    model = pickle.load(open(argv[1], 'rb'))
    testing_file = argv[2]
    prediction = argv[3]
    
    with open(testing_file, 'rb') as f:
        testing_data = np.array(list(csv.reader(f))).astype(float)

    labels = predict(model, testing_data)
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
