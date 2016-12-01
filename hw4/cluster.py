# Reference: http://scikit-learn.org/stable/auto_examples/text/document_clustering.html
from __future__ import print_function
import sys, csv, string, codecs, logging
import numpy as np
import cPickle as pickle
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans

stoplist = list(csv.reader( open('stop-word-list.csv')))[0]
replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))

def run(argv):
    dir_path = argv[1]
    outfile = argv[2]
    
    logging.basicConfig( format='%(levelname)s : %(message)s', level=logging.INFO)

    print('Loading title_StackOverflow.txt ... ')
    sentences = []
    with open(dir_path+'title_StackOverflow.txt', 'r') as f:
        for line in f:
            line = line.translate(replace_punctuation) 
            line = line.decode('utf-8').encode('ascii', 'ignore').lower().split()
            words = [word for word in line if word not in stoplist] 
            sentences.append( ' '.join(words) )

    print("Extracting features from the training dataset using a sparse vectorizer")
    vectorizer = TfidfVectorizer( max_df = 0.5, 
                                  min_df = 2,
                                  #stop_words=stoplist,
                                  use_idf=True,
                                  sublinear_tf=True
                                  )
    X = vectorizer.fit_transform( sentences )

    
    print("n_samples: %d, n_features: %d" % X.shape)
    
    print("Performing dimensionality reduction using LSA")
    svd = TruncatedSVD(20) #TODO
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    explained_variance = svd.explained_variance_ratio_.sum()
    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))


    print("KMeans Clustering ... ")
    MINI = True
    true_k = 35 #TODO
    if MINI:
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=1)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                    verbose=1)
    km.fit(X)


    print("Top terms per cluster:")
    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()

    labels = km.labels_

    predictions = np.zeros(5000000)
    with open(dir_path+'check_index.csv') as f:
        incsv = csv.reader(f)
        next(incsv)
        for row in incsv:
            if labels[int(row[1])] == labels[int(row[2])]:
                predictions[ int(row[0]) ] = 1
    
    with open(outfile, 'wb') as f:
        f.write('ID,Ans\n')
        for i in xrange(predictions.shape[0]):
            f.write('%d,%d\n'%(i,predictions[i]))

if __name__ == '__main__':
    if len(sys.argv) == 3:
        run(sys.argv)
    else:
        print('INPUT FORMAT ERROR!')
