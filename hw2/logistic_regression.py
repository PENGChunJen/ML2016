import sys, csv
import numpy as np
import cPickle as pickle


def train_model(training_data):
    model = training_data 
    return model


def predict(model, data):
    #labels = []
    labels = np.zeros(600)
    return labels


def train(argv):
    training_file = argv[1]
    output_model = argv[2]
    
    with open(training_file, 'rb') as f:
        training_data = list(csv.reader(f))
    
    model = train_model( training_data ) 

    with open(output_model, 'wb') as f:
        pickle.dump( model, f )


def test(argv):
    model = pickle.load(open(argv[1], 'rb'))
    testing_file = argv[2]
    prediction = argv[3]
    
    with open(testing_file, 'rb') as f:
        testing_data = list(csv.reader(f))
    
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
