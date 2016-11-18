import sys
import numpy as np
import cPickle as pickle
from keras.models import Sequential, Model, load_model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from sklearn.model_selection import cross_val_score, KFold
from sklearn.utils import shuffle

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


def load_unlabel_data (file_name):
    all_unlabel = np.array(pickle.load(open(file_name, 'rb')))
    X_unlabel = np.empty((all_unlabel.shape[0], 32, 32, 3), dtype="float32")
    for i in xrange(all_unlabel.shape[0]):
        X_unlabel[i] = all_unlabel[i].reshape(3,32,32).transpose(1,2,0)/255.
    return X_unlabel


def add_train_data(X_train, Y_train, X_unlabel, Y_unlabel, threshold):
    labels = np.argmax( Y_unlabel, axis=1 )
    print ''
    for i in xrange(10):
        examples = np.where( np.amax( Y_unlabel, axis=1 )  > i/10. )
        print examples[0].shape[0], 'unlabeled data above threshold:', i/10. 

    examples = np.where( np.amax( Y_unlabel, axis=1 )  > threshold )
    print examples[0].shape[0], 'unlabeled data above threshold:', threshold

    a = np.zeros(Y_unlabel.shape)
    a[ examples, labels[examples] ] = 1

    X_new = X_unlabel[examples]
    Y_new = a[examples]

    X_new_train = np.concatenate((X_train, X_new))
    Y_new_train = np.concatenate((Y_train, Y_new))

    return shuffle(X_new_train, Y_new_train)
   

def train_model(X_train, Y_train, X_val, Y_val, opt, callbacks, model_weight_name = None):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode = 'same', 
                            input_shape = (X_train.shape[1:])))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode = 'same')) 
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3)) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(output_dim = 512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim = 10))
    model.add(Activation('softmax'))

    if model_weight_name is not None:
        model.load_weights(model_weight_name)


    model.compile(loss = 'categorical_crossentropy', 
                  optimizer = opt, metrics=['accuracy'])


    model.fit( X_train, Y_train,
               batch_size=32, 
               nb_epoch=100,
               shuffle=True,
               verbose=1,
               validation_data = (X_val, Y_val),
               callbacks = callbacks)
               #validation_split = 0.1 )
    return model


def train(argv):
    dir_path = argv[1]
    model_name = argv[2]

    #sgd = SGD(lr = 0.01, decay = 1e-6, momentum=0.9, nesterov = True)
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    adam = Adam(lr=0.001, beta_1 = 0.9, beta_2=0.999, epsilon=1e-0, decay=0.0)

    earlystopping = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1, mode='auto') 
    checkpointer = ModelCheckpoint( filepath = './weights.hdf5', verbose = 0, save_best_only = True)
    tensorboard = TensorBoard( log_dir = './logs/init', histogram_freq = 0, write_graph=True, write_images = False)
    callbacks = [tensorboard, earlystopping, checkpointer]

    print 'Loading all_label.p ...'
    X_train, Y_train, X_val, Y_val = load_label_data(dir_path+'all_label.p')
    #model = train_model(X_train, Y_train, X_val, Y_val, adadelta, callbacks, 'weights.hdf5')
    model = train_model(X_train, Y_train, X_val, Y_val, adadelta, callbacks)
    model.save(model_name)

    print 'Loading all_unlabel.p ...'
    X_unlabel = load_unlabel_data(dir_path+'all_unlabel.p')
    repeat = 10 
    for iteration in xrange(repeat):
        print 'Predicting all_unlabel.p ...'
        Y_unlabel = model.predict(X_unlabel, batch_size=256, verbose=1)
        threshold = 0.8
        X_new_train, Y_new_train = add_train_data(X_train, Y_train, X_unlabel, Y_unlabel, threshold)
        tensorboard = TensorBoard( log_dir = './logs/iter%s'%iteration, histogram_freq = 0, write_graph=True, write_images = False)
        earlystopping = EarlyStopping(monitor='val_acc', min_delta = 0, patience = 10, verbose = 1, mode='auto') 
        callbacks = [tensorboard, earlystopping, checkpointer]
        model = train_model(X_new_train, Y_new_train, X_val, Y_val, adadelta, callbacks, 'weights.hdf5')
        model.save(model_name)


def test(argv):
    dir_path = argv[1]
    model_name = argv[2]
    #model_name = 'model0.h5'
    prediction = argv[3]
    
    print 'Loading test.p ...'
    # test['data'][i(0-9999)] = [1024, 1024, 1024]
    test = pickle.load(open(dir_path+'test.p', 'rb'))
    all_unlabel = np.array(test['data'])
    X_unlabel = np.empty((all_unlabel.shape[0], 32, 32, 3), dtype="float32")
    for i in xrange(all_unlabel.shape[0]):
        X_unlabel[i] = all_unlabel[i].reshape(3,32,32).transpose(1,2,0)/255.

    print 'Predicting test.p ...'
    model = load_model(model_name)
    Y_unlabel = model.predict(X_unlabel, batch_size=32, verbose=1)
    labels = np.argmax(Y_unlabel, axis=1)
    
    with open(prediction, 'wb') as f:
        f.write('ID,class\n')
        for i in xrange(labels.shape[0]):
            f.write('%d,%d\n'%(i,labels[i]))

if __name__ == '__main__':
    if len(sys.argv) == 3:
        train(sys.argv)
    elif len(sys.argv) == 4:
        test(sys.argv)
    else:
        print 'INPUT FORMAT ERROR!'
        print 'Usage 1: ./train.sh <training data> <output model>'
        print 'Usage 2: ./test.sh <model name> <testing data> prediction.csv'
