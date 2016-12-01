import sys
import numpy as np

def load_label_data (file_name):
    all_label = np.array(pickle.load(open(file_name, 'rb')))
    X_label = np.empty((5000, 32, 32, 3), dtype="float32")
    Y_label = np.zeros((5000,10))
    num = 0
    for i in xrange(all_label.shape[0]):
        for j in xrange(all_label.shape[1]):
            X_label[num] = all_label[i][j].reshape(3,32,32).transpose(1,2,0)/255.
            Y_label[num][i] = 1 
            num = num+1

    X_label, Y_label = shuffle(X_label, Y_label)
    num_val = 1000
    X_val, Y_val = X_label[:num_val], Y_label[:num_val]
    X_train, Y_train = X_label[num_val:], Y_label[num_val:]

    return X_train, Y_train, X_val, Y_val

def run(argv):
    dir_path = argv[1]
    prediction = argv[2]
    
    titles = open(dir_path+'title_StackOverflow.txt')
    X_train, Y_train, X_val, Y_val = load_label_data(dir_path+'check_index.csv')
    docs = open(dir_path+'docs.txt')

    labels = np.zeros(20000)
    
    with open(prediction, 'wb') as f:
        f.write('ID,Ans\n')
        for i in xrange(labels.shape[0]):
            f.write('%d,%d\n'%(i,labels[i]))

if __name__ == '__main__':
    if len(sys.argv) == 3:
        run(sys.argv)
    else:
        print 'INPUT FORMAT ERROR!'
