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
        if s == 0:
            s = m
            m = 0
        mean_r.append(m)
        std_r.append(s)
        X_norm[:,i] = (X_norm[:,i]-m)/s
    return X_norm, np.array([mean_r]), np.array([std_r])


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

def gradient_descent_v1(X, Y, theta, learning_rate, iterations):
    num_points, num_features = X.shape
    for i in xrange(iterations):
        hypothesis = X.dot(theta)
        loss = hypothesis - Y
        gradient = np.dot(X.T, loss)/num_points
        print theta.shape, gradient.shape
        theta = theta - learning_rate * gradient
        
        cost = loss.T.dot(loss) / (2.*num_points)
        #cost = np.sum(loss ** 2) / (2*num_points)
        if i%1000 == 0 :
            print("Iteration %5d | Cost: %f" %(i, cost))
    
    return theta, cost

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
    
    return theta, cost

def compute_cost(X, Y, theta):
    m = Y.size
    error = X.dot(theta) - Y
    J = (1.0 /(2*m)) * error.T.dot(error)
    return J

def split_to_N_fold(X, Y, fold):
    num_points, num_features = X.shape
    batch_size = num_points / fold 
    batch = [] 
    for i in xrange(fold):
        validation_data = X[batch_size*i:batch_size*(i+1),:] 
        validation_value = Y[batch_size*i:batch_size*(i+1)]
        train_data = np.concatenate((X[:(batch_size*i),:],X[batch_size*(i+1):,:]))
        train_value = np.concatenate((Y[:batch_size*i],Y[batch_size*(i+1):]))
        batch.append((train_data, train_value, validation_data, validation_value)) 
    return batch 

def run_gradient_descent(train_X, train_Y, validation_X, validation_Y, theta, learning_rate):
    num_points, num_features = train_X.shape
    hypothesis = train_X.dot(theta)
    loss = hypothesis - train_Y
    gradient = np.dot(train_X.T, loss)/num_points
    theta = theta - learning_rate * gradient
    train_cost = loss.T.dot(loss) / (2.*num_points)

    validation_cost = compute_cost(validation_X, validation_Y, theta)
    return theta, train_cost, validation_cost

def gradient_descent(train_X, train_Y, validation_X, validation_Y, theta, learning_rate=0.01, iterations=10001):
    #num_points, num_features = train_X.shape
    delta = 0.000001
    last_cost = float('Inf')
    final_it = 0
    for it in xrange(iterations):
        theta, train_cost, validation_cost = run_gradient_descent(train_X, train_Y, validation_X, validation_Y, theta, learning_rate)
        
        if it%1000 == 0 :
            #print("Iteration %5d | Train Cost: %f | Validation Cost: %f " %(it, train_cost, validation_cost))
            #print theta
            if abs(last_cost - validation_cost) < delta:
                #print 'last_cost:', last_cost, 'cost:', validation_cost
                #print("Converge at Iteration %5d | Train Cost: %f | Validation Cost: %f " %(it, train_cost, validation_cost))
                break

        if it > 1000 and validation_cost > last_cost:
            #print "Rebound"
            #print("Iteration %5d | Train Cost: %f | Validation Cost: %f " %(it, train_cost, validation_cost))
            final_it = it
            break

        
        last_cost = validation_cost

    
    return theta, train_cost, validation_cost, final_it

def cross_validation(X, Y, learning_rate, iterations, fold = 5):
    num_points, num_features = X.shape
    if fold == -1:
        validation_data = X[:num_points/10,:] 
        validation_value = Y[:num_points/10]
        train_data = X[num_points/10:,:]
        train_value = Y[num_points/10:]
        batch = [(train_data, train_value, validation_data, validation_value)]
    else: 
        batch = split_to_N_fold(X, Y, fold)
    avg_theta = np.zeros(num_features)
    avg_cost = 0.
    for i in xrange(len(batch)):
        theta = np.zeros(num_features)
        
        train_X, train_Y, validation_X, validation_Y = batch[i]
        #print X.shape, train_X.shape, validation_X.shape
        #print Y.shape, train_Y.shape, validation_Y.shape 

        theta, train_cost, validation_cost, final_it = gradient_descent(train_X, train_Y, validation_X, validation_Y, theta, learning_rate, iterations)
        avg_theta += theta     
        avg_cost += validation_cost
    #print 'Average cost:', avg_cost/fold
    theta = avg_theta/fold
    return theta, avg_cost

def expand_as_polynomial(X, test_X, p =  2):
    num_points, num_features = X.shape
    X_poly = np.array([np.ones(num_points)]).T
    num_points, num_features = test_X.shape
    test_X_poly = np.array([np.ones(num_points)]).T

    for i in xrange(1,p+1):
        X_poly = np.concatenate((X_poly, np.power(X,i)), axis=1)
        test_X_poly = np.concatenate((test_X_poly, np.power(test_X,i)), axis=1)

    X, mean_r, std_r = normalize_feature(X_poly)
    test_X = (test_X_poly - mean_r)/std_r
    
    return X, test_X

def sequential_forward_selection(X, Y, test_X):
    # Parameter setting
    max_iterations = 1000001
    learning_rate = 0.01
    fold = -1
    selected_features = [] 
    num_points, num_features = X.shape
    last_cost = float('INF') 
    cost = 1000000000
    while last_cost > cost:
        last_cost = cost
        feature_costs = [] 
        #for i in xrange(num_features):
        for i in xrange(18):
            if i in selected_features: continue
            f = [k for k in selected_features]
            f.append(i)
            
            ff = []
            for row_f in f:
                ff.extend(range(9*row_f,9*(row_f+1)))
            print 'ff:', ff 
            '''
            if len(f) == 1:
                feature_X = X[:,i].reshape(num_points,1)
            else:
                feature_X = X[:,f]
            '''
            feature_X = X[:,ff]
            print feature_X.shape

            theta, cost = cross_validation(feature_X, Y, learning_rate, max_iterations, fold)
            feature_costs.append({'feature':f, 'cost':cost})
            print 'feature:',f, 'cost:',cost
    
        sorted_list = sorted(feature_costs, key = lambda k: k['cost']) 
        cost = sorted_list[0]['cost']
        selected_features = sorted_list[0]['feature']
        print 'feature:', selected_features, 'cost:',cost
    return X, test_X

def run():
    # Parameter setting
    max_iterations = 100001
    learning_rate = 0.01
    fold = 5

    X, Y = readTrainingData()
    test_X = readTestingData()
    X, test_X = expand_as_polynomial(X, test_X, 1)
    X, test_X = sequential_forward_selection(X, Y, test_X)

    theta = cross_validation(X, Y, learning_rate, max_iterations, fold)
   
    predict = (test_X).dot(theta)
    write_to_file(predict, "kaggle_best.csv")

if __name__ == '__main__':
    run()







