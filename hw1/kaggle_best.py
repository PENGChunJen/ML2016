import numpy as np
import csv
#import cma
import pickle
#from matplotlib import pyplot

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
    x = np.array(X).astype(np.float)#,np.array([[1,2],[4,5],[7,8]])
    y = np.array(Y).astype(np.float)#np.array([3,6,9])
   
    c = np.concatenate((x,y.reshape(1,y.shape[0]).T), axis=1)
    np.random.shuffle(c)
    x = c[:,:-1]
    y = c[:,-1:].flatten()
    return x, y
    #return np.array(X).astype(np.float), np.array(Y).astype(np.float)

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
    delta = 1e-11
    max_explore = float('Inf') 
    #max_explore = 100
    cost_save_point = float('Inf')
    final_it = 0
    count = 0
    for it in xrange(iterations):
        theta, train_cost, validation_cost = run_gradient_descent(train_X, train_Y, validation_X, validation_Y, theta, learning_rate)
        
        if abs(cost_save_point - validation_cost) < delta or validation_cost > 1000000:
            print("Iteration %5d | Train Cost: %.10f | Validation Cost: %.10f " %(it, train_cost, validation_cost))
            final_it = it
            break
        
        if it%1000 == 0 :
            print("Iteration %5d | Train Cost: %.10f | Validation Cost: %.10f | count: %d" %(it, train_cost, validation_cost, count))
            if count > max_explore:
                count = 0
                learning_rate = learning_rate/10
                print('Adjust learning_rate to %.10f' %(learning_rate))
            elif validation_cost > cost_save_point :
                learning_rate = learning_rate/10
                theta = theta_save_point
                count = 0
                print('Adjust learning_rate to %.10f, theta row back to iteration %d' %(learning_rate, it-1000))
            else:
                count += 1 
                theta_save_point = theta
            cost_save_point = validation_cost
    
    return theta, train_cost, validation_cost, final_it

def cross_validation(X, Y, learning_rate, iterations, fold = 5):
    num_points, num_features = X.shape
    if fold == -1:
        validation_data = X[:num_points/10,:] 
        validation_value = Y[:num_points/10]
        train_data = X[num_points/10:,:]
        train_value = Y[num_points/10:]
        #validation_data = X[:500,:] 
        #validation_value = Y[:500]
        #train_data = X[500:,:]
        #train_value = Y[500:]
        batch = [(train_data, train_value, validation_data, validation_value)]
    else: 
        batch = split_to_N_fold(X, Y, fold)
    avg_theta = np.zeros(num_features)
    avg_cost = 0.
    for i in xrange(len(batch)):
        theta = np.zeros(num_features)
        #theta = np.random.rand(num_features)
        
        train_X, train_Y, validation_X, validation_Y = batch[i]
        #print X.shape, train_X.shape, validation_X.shape
        #print Y.shape, train_Y.shape, validation_Y.shape 

        theta, train_cost, validation_cost, final_it = gradient_descent(train_X, train_Y, validation_X, validation_Y, theta, learning_rate, iterations)
        print 'Fold', i, ': cost:', validation_cost
        avg_theta += theta     
        avg_cost += validation_cost
        if fold == -1:
            return theta, validation_cost

    #print fold, 'folds cross-validation average cost:', avg_cost/fold
    theta = avg_theta/fold
    avg_cost = avg_cost/fold
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

def sequential_forward_selection(X, Y, test_X, features = None):
    if features is not None:
        ff = []
        for row_f in features:
            ff.extend(range(9*row_f,9*(row_f+1)))
        return X[:,ff], test_X[:,ff]

    # Parameter setting
    max_iterations = 1000001
    learning_rate = 0.01
    fold = 10 
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
            #print 'ff:', ff 
            '''
            if len(f) == 1:
                feature_X = X[:,i].reshape(num_points,1)
            else:
                feature_X = X[:,f]
            '''
            feature_X = X[:,ff]
            #print feature_X.shape

            theta, cost = cross_validation(feature_X, Y, learning_rate, max_iterations, fold)
            feature_costs.append({'feature':f, 'cost':cost})
            print 'feature:',f, 'cost:',cost
    
        sorted_list = sorted(feature_costs, key = lambda k: k['cost']) 
        cost = sorted_list[0]['cost']
        selected_features = sorted_list[0]['feature']
        print '\nCurrently selected inputs: feature:', selected_features, 'cost:',cost, '\n'
    return X, test_X

def run():
    #print 'learning_rate:',learning_rate
    # Parameter setting
    max_iterations = 10000001
    learning_rate = 0.01
    fold = 10

    X, Y = readTrainingData()
    test_X = readTestingData()
    X, test_X = expand_as_polynomial(X, test_X, 1) # add bias, poly = 0
    #X, test_X = expand_as_polynomial(X, test_X, poly)

    # SFS
    #selected_features = None
    #selected_X, selected_test_X = sequential_forward_selection(X, Y, test_X, selected_features)

    # 10 fold cross-validation feature selection
    #selected_features = [0,10,9,13,12,4,2,8,3,17]
    #while len(selected_features) > 0:
    #    selected_X, selected_test_X = sequential_forward_selection(X, Y, test_X, selected_features)
    #    avg_theta, avg_cost = cross_validation(selected_X, Y, learning_rate, max_iterations, fold)
    #    print '10 fold cross-validation cost:', avg_cost, selected_features
    #    selected_features.pop()
     
    #selected_features = [0,10,9,13,12,4] # always need 0 as bias
    selected_features = [0,6,8,9,10]
    selected_X, selected_test_X = sequential_forward_selection(X, Y, test_X, selected_features)

    avg_theta, avg_cost = cross_validation(selected_X, Y, learning_rate, max_iterations, fold)
    print '10 fold cross-validation cost:', avg_cost, selected_features
    predict = (selected_test_X).dot(avg_theta)
    #write_to_file(predict, ""+str(avg_cost)+"_"+str(learning_rate)+".csv")
    write_to_file(predict, "kaggle_best.csv")
    return avg_cost

def plot_learning_rate():
    learning_rate = np.arange(0.001,0.1,0.005)
    cost_history = []
    data = []
    #data = pickle.load(open('learning_rate', 'rb')) 
    for l in learning_rate:
        cost = run(l)
        cost_history.append(cost)
        data.append({'learning_rate':l, 'cost':cost})
    pickle.dump(data, open('new_feature_learning_rate', 'wb'))
    #pyplot.plot(learning_rate, cost_history) 
    #pyplot.xlabel('learning rate')
    #pyplot.ylabel('loss')
    #pyplot.savefig('tuning_learning_rate1.png')

def run_CMAES(x, *args):
    learning_rate_history = []
    cost_history = []
    for learning_rate in x:
        cost.append(run(learning_rate))
    return cost_history 
    
if __name__ == '__main__':
   
    run()
    #plot_learning_rate()
    '''
    res = cma.fmin(run_CMAES, 2*[0.01], 0.01)
    print res[0]
    cma.plot()
    cma.savefig('cmaes_tune_learningrate.png')
    cma.closefig()

    cdl = cma.CMADataLoger().downsampling().plot()
    '''
