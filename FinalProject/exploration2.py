import sys
import pickle
import math
import matplotlib.pyplot as plt

from time import time

from dtypes import dtypes, ind, day_mapping, region_mapping

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn import linear_model
from sklearn import decomposition
from sklearn import cross_validation as cv
from sklearn import grid_search as gs
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
import sklearn.preprocessing as preprocessing
import sklearn.feature_selection as fs


import numpy as np
import scipy

import load_data

import logging
logging.basicConfig(filename='final_model.log',level=logging.DEBUG)


# from read_csv import read_csv_to_numpy_array
model_directory = '/Users/rjohnson/Documents/DS/DataScience/FinalProject/models/'


def subjectKMeans(subjects, labels):
    #Set up Text Vecor (many options here)
  true_k = np.unique(labels).shape[0]
  vectorizer = TfidfVectorizer(max_df=0.5, 
                                  stop_words='english', use_idf=True,
                                  encoding='unicode', norm='l1',
                                  lowercase=True, strip_accents='unicode')
  X = vectorizer.fit_transform(subjects)

  #Some possible Dimensionality Reduction
  t0 = time()
  lsa = TruncatedSVD(10)
  X = lsa.fit_transform(X)
  # Vectorizer results are normalized, which makes KMeans behave as
  # spherical k-means for better results. Since LSA/SVD results are
  # not normalized, we have to redo the normalization.
  X = Normalizer(copy=False).fit_transform(X)

  print("done in %fs" % (time() - t0))
  print()

  # Do the actual clustering
  km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=True)

  print("Clustering sparse data with %s" % km)
  t0 = time()
  km.fit(X)
  print("done in %0.3fs" % (time() - t0))
  print()

  print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
  print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
  print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
  print("Adjusted Rand-Index: %.3f"
        % metrics.adjusted_rand_score(labels, km.labels_))
  
  # I Can't do this for some reason...
  # print("Silhouette Coefficient: %0.3f"
  #       % metrics.silhouette_score(X, labels, sample_size=1000))

  print()


      
def segment_for_even_distribution(train_set, train_labels):
  indexes = range(0, len(train_labels))
  newLabels = train_labels[indexes]
  index_of_indexes = []
  so_far = 0
  total = sum([1 if label == 0 else 0 for label in train_labels])
  for i, row in enumerate(newLabels):
    if newLabels[i] == 1 and so_far < total:
      index_of_indexes.append(i)
      so_far += 1
    elif newLabels[i] == 0:
      index_of_indexes.append(i)
  final_indexes = [indexes[i] for i in index_of_indexes]
  return train_set[final_indexes], train_labels[final_indexes]

def plot_distribution(data, index, title):
  col = select_data(data, [index])
  plt.hist(col)
  plt.title(title)
  plt.show()

def decomposition_pca(train, test):
  """ Linear dimensionality reduction """
  pca = decomposition.PCA(n_components=12, whiten=True)
  train_pca = pca.fit_transform(train)
  test_pca = pca.transform(test)
  return train_pca, test_pca

def decomposition_pca_train(train):
  """ Linear dimensionality reduction """
  pca = decomposition.PCA(n_components=12, whiten=True)
  train_pca = pca.fit_transform(train)
  return train_pca

def split_data(X_data, y_data):
  """ Split the dataset in train and test """
  return cv.train_test_split(X_data, y_data, test_size=0.1, random_state=0)

def grid_search(y_data):
  c_range = 10.0 ** np.arange(6.5,7.5,.25)
  gamma_range = 10.0 ** np.arange(-1.5,0.5,.25)
  params = [{'kernel': ['linear'], 'gamma': gamma_range, 'C': c_range}]

  cvk = cv.StratifiedKFold(y_data, n_folds=5)
  return gs.GridSearchCV(svm.SVC(), params, cv=cvk)

def train(features, result):
  """ Use features and result to train Support Vector Machine"""
  clf = grid_search(result)
  logging.info("Gonna start fitting something")
  start = time()
  clf.fit(features, result)
  logging.info("Fit took this long:")
  logging.info(time() - start)
  return clf

def predict(clf, features):
  """ Predict labels from trained CLF """
  return clf.predict(features)

def show_score(clf, X_test, y_test):
  """ Scores are computed on the test set """
  y_pred = predict(clf, X_test)
  logging.info(metrics.classification_report(y_test, y_pred))

def write_data(filename, data):
  """ Write numpy array into CSV """
  np.savetxt(filename, data, fmt='%d')

def normalize_data(x_data):
  min_max_scaler = preprocessing.MinMaxScaler()
  category_data = select_data(x_data, [0,1])
  numerical_data = select_data(x_data, range(2,45-offset))
  x_data = np.hstack((category_data, numerical_data)) 


def svmTrainAndPrintScore(training_set, labels, test_set, test_labels):
  clf = svm.SVC()
  clf.fit(training_set, labels)
  logging.info("-------- SVM via svm.svc no params  ------------")
  logging.info("--------PERFORMANCE ON TRAINING DATA------------")
  show_score(clf, training_set, labels)
  logging.info("----------PERFORMANCE ON TEST DATA--------------")
  show_score(clf, test_set, test_labels)

def logRegTrainAndPrintScore(trainer, labels, test, test_labels, c_range=[1], penalties=['l2'], chi2_features=None):
  if chi2_features:
    len_training=len(labels)
    test_train = scipy.sparse.vstack([trainer, test])
    test_train = scipy.sparse.csr_matrix(test_train)
    all_labels = np.append(labels, test_labels)
    chi2 = fs.chi2(test_train, all_labels)
    strong = []
    weak = []
    for i, p in enumerate(chi2[1]):
      if p < 0.1:
        weak.append(i)
        if p < .05:
          strong.append(i)
    strongFeatures = test_train[:,strong]
    weakFeatures = test_train[:,weak]
    if chi2_features == 'weak':
      trainer = weakFeatures[:len_training]
      test = weakFeatures[len_training:]
    else:
      trainer = strongFeatures[:len_training]
      test = strongFeatures[len_training:]

  for c in c_range:
    for penalty in penalties:
      clf = linear_model.LogisticRegression(C=c, penalty=penalty)
      clf.fit(trainer, labels)
      logging.info("-------- LogisticRegression via linear_model.LogisticRegression C =%f, Penalty = %s ------------" % (c, penalty))
      logging.info("--------PERFORMANCE ON TRAINING DATA------------")
      show_score(clf, trainer, labels)
      logging.info("----------PERFORMANCE ON TEST DATA--------------")
      show_score(clf, test, test_labels)

def svmTrainandPrintWithGridSearch(training_set, labels, test_set, test_labels):
  clf = train(training_set, labels)
  logging.info(show_score(clf, test_set, test_labels))


if __name__ == "__main__":
  # request_data = read_csv_to_numpy_array(data_directory + 'request_info_train.txt')
  np.random.seed(9)

  # if len(sys.argv) > 1:
  #   build_data = int(sys.argv[1])
  # else:
  #   print "you didn't say whether or not to build the data so we'll pickle it"
  #   build_data=0
  
  # PLOTTING SOME DATA
  # for i, item in enumerate(X_train):
  # # plot_distribution(X_train, 0, "Day")
  
  vectorizer = 'TfidfVectorizer'
  stem = 'RegexpStemmer'
  v2 = 'HashingVectorizer'
  # v2 = 'TfidfVectorizer' # To work with NOrmalization
  n1 = '1000'
  n2 = '100000'
  strong_ext = '_chi2_strong_' + vectorizer
  weak_ext = '_chi2_weak_' + vectorizer
  train_features, train_labels = load_data.load_feature_data(0, test_train='train')
  len_train= len(train_labels)
  test_features, test_labels = load_data.load_feature_data(0, test_train='test')
  all_labels = np.append(train_labels, test_labels)
  ## Do better Normalization on Customer Features
  all_features = np.vstack((train_features, test_features))
  float_feats = [[float(i) for i in row] for row in all_features]  # Turn values to floating point
  new_feats = preprocessing.normalize(float_feats, norm='l1', axis=0)
  train_features = new_feats[:len_train]
  test_features = new_feats[len_train:]
  ### Done Normalizing
  train_email_features, test_email_features = load_data.load_email_data(0, vectorizer, stemmer=stem, vectorizer=v2)
  train_subject_features, test_subject_features = load_data.load_subject_data(0, vectorizer, stemmer=stem, vectorizer=v2)
  train_email_features_n1, test_email_features_n1 = load_data.load_email_data(0, n1+v2, stemmer=stem, vectorizer=v2)
  train_subject_features_n1, test_subject_features_n1 = load_data.load_subject_data(0, n1+v2, stemmer=stem, vectorizer=v2)
  train_email_features_n2, test_email_features_n2 = load_data.load_email_data(0, n2+v2, stemmer=stem, vectorizer=v2)
  train_subject_features_n2, test_subject_features_n2 = load_data.load_subject_data(0, n2+v2, stemmer=stem, vectorizer=v2)
  train_email_strong_features, test_email_strong_features = load_data.load_email_data(0, strong_ext, stemmer=stem, vectorizer=vectorizer)
  train_subject_strong_features, test_subject_strong_features = load_data.load_subject_data(0, strong_ext, stemmer=stem, vectorizer=vectorizer)
  train_email_weak_features, test_email_weak_features = load_data.load_email_data(0, weak_ext, stemmer=stem, vectorizer=vectorizer)
  train_subject_weak_features, test_subject_weak_features = load_data.load_subject_data(0, weak_ext, stemmer=stem, vectorizer=vectorizer)

  logging.info("All Data Loaded")
  for t in ['email']:
    for s in ['n2']:
      if t == 'email':
        if s =='strong':
          trainer = train_email_strong_features
          test = test_email_strong_features
        elif s == 'weak':
          trainer = train_email_weak_features
          test = test_email_weak_features
        elif s == 'n1':
          trainer = train_email_features_n1
          test = test_email_features_n1
        elif s == 'n2':
          trainer = train_email_features_n2
          test = test_email_features_n2
        else:
          trainer = train_email_features
          test = test_email_features
        split_train, split_labels = segment_for_even_distribution(trainer, train_labels)
        logging.info("============================================================")
        logging.info("Grid Search SVM on Email Text with %s features" % s)
        # svmTrainAndPrintScore(split_train, split_labels, test, test_labels)
        logRegTrainAndPrintScore(split_train, split_labels, test, test_labels)        
      elif t == 'subject':
        if s =='strong':
          trainer = train_subject_strong_features
          test = test_subject_strong_features
        elif s == 'weak':
          trainer = train_subject_weak_features
          test = test_subject_weak_features
        elif s == 'n1':
          trainer = train_subject_features_n1
          test = test_subject_features_n1
        elif s == 'n2':
          trainer = train_subject_features_n2
          test = test_subject_features_n2
        else:
          trainer = train_subject_features
          test = test_subject_features
        split_train, split_labels = segment_for_even_distribution(trainer, train_labels)
        logging.info("============================================================")
        logging.info("Grid Search SVM on Subject Text with %s features" % s)
        # svmTrainAndPrintScore(split_train, split_labels, test, test_labels)
        logRegTrainAndPrintScore(split_train, split_labels, test, test_labels)        
      elif t == 'both':
        if s =='strong':
          trainer_s = train_subject_strong_features
          test_s = test_subject_strong_features
          trainer_e = train_email_strong_features
          test_e = test_email_strong_features
        elif s == 'weak':
          trainer_s = train_subject_weak_features
          test_s = test_subject_weak_features
          trainer_e = train_email_weak_features
          test_e = test_email_weak_features
        elif s =='n1':
          trainer_s = train_subject_features_n1
          test_s = test_subject_features_n1
          trainer_e = train_email_features_n1
          test_e = test_email_features_n1
        elif s =='n2':
          trainer_s = train_subject_features_n2
          test_s = test_subject_features_n2
          trainer_e = train_email_features_n2
          test_e = test_email_features_n2
        else:
          trainer_s = train_subject_features
          test_s = test_subject_features
          trainer_e = train_email_features
          test_e = test_email_features
        trainer = scipy.sparse.hstack([trainer_s, trainer_e])
        test = scipy.sparse.hstack([test_s, test_e])
        trainer = scipy.sparse.csr_matrix(trainer)
        test = scipy.sparse.csr_matrix(test)
        split_train, split_labels = segment_for_even_distribution(trainer, train_labels)
        logging.info("============================================================")
        logging.info("Grid Search SVM on ALL Text with %s features" % s)
        # svmTrainAndPrintScore(split_train, split_labels, test, test_labels)
        logRegTrainAndPrintScore(split_train, split_labels, test, test_labels)        
      elif t =='normal':
        trainer = train_features
        test = test_features
        split_train, split_labels = segment_for_even_distribution(trainer, train_labels)
        logging.info("============================================================")
        logging.info("Grid Search SVM on Customer Info with %s features" % s)
        # svmTrainAndPrintScore(split_train, split_labels, test, test_labels)
        logRegTrainAndPrintScore(split_train, split_labels, test, test_labels)        
      elif t == 'all':
        if s =='strong':
          trainer_s = train_subject_strong_features
          test_s = test_subject_strong_features
          trainer_e = train_email_strong_features
          test_e = test_email_strong_features
        elif s == 'weak':
          trainer_s = train_subject_weak_features
          test_s = test_subject_weak_features
          trainer_e = train_email_weak_features
          test_e = test_email_weak_features
        elif s =='n1':
          trainer_s = train_subject_features_n1
          test_s = test_subject_features_n1
          trainer_e = train_email_features_n1
          test_e = test_email_features_n1
        elif s =='n2':
          trainer_s = train_subject_features_n2
          test_s = test_subject_features_n2
          trainer_e = train_email_features_n2
          test_e = test_email_features_n2
        else:
          trainer_s = train_subject_features
          test_s = test_subject_features
          trainer_e = train_email_features
          test_e = test_email_features
        train_features = [[abs(i) for i in row] for row in train_features]
        test_features = [[abs(i) for i in row] for row in test_features]
        train_feats_sparse = scipy.sparse.csr_matrix(train_features)
        test_feats_sparse = scipy.sparse.csr_matrix(test_features)
        trainer = scipy.sparse.hstack([train_feats_sparse, trainer_s, trainer_e])
        test = scipy.sparse.hstack([test_feats_sparse, test_s, test_e])
        trainer = scipy.sparse.csr_matrix(trainer)
        test = scipy.sparse.csr_matrix(test)
        split_train, split_labels = segment_for_even_distribution(trainer, train_labels)
        logging.info("============================================================")
        logging.info("Grid Search SVM on ALL DATA with %s features" % s)
        # svmTrainAndPrintScore(split_train, split_labels, test, test_labels)
        # c_range = [.1, 1, 10 ,100, 1000, 10000]
        # penalties = ['l1', 'l2']
        logRegTrainAndPrintScore(split_train, split_labels, test, test_labels)        

  # #SPlit up the data to better parts
  # for stem in ['PorterStemmer', 'RegexpStemmer', 'LancasterStemmer']:
  #   for vectorizer in ['TfidfVectorizer', 'HashingVectorizer']:
  #     logging.info("Loading Data for %s, %s:" % (stem, vectorizer))
  #     train_email_features, test_email_features = load_data.load_email_data(0, vectorizer, stemmer=stem, vectorizer=vectorizer)
  #     train_subject_features, test_subject_features = load_data.load_subject_data(0, vectorizer, stemmer=stem, vectorizer=vectorizer)
  #     for textType in ['email', 'subject']:
  #       logging.info("Building Model for %s " % textType)
  #       if textType == 'email':
  #         train, test = train_email_features, test_email_features
  #       elif textType == 'subject':
  #         train, test = train_subject_features, test_subject_features
  #       # else:
  #       #   train = np.hstack((train_subject_features, train_email_features))
  #       #   test = np.hstack((test_subject_features, test_email_features))

  #       split_train, split_labels = segment_for_even_distribution(train, train_labels)
  #       logging.info("Data Split, Starting Model Build for %s, %s with length %i at: " % (stem, vectorizer, len(split_labels)))
  #       t0 = time()
  #       svmTrainAndPrintScore(split_train, split_labels, test, test_labels)
  #       logging.info("done in %0.3fs" % (time() - t0))
  #       logging.info("======================================================================================================")


   # X_data, y_data = load_feature_data(build_data=build_data, test_train='train')
  # test_data, test_labels = load_feature_data(build_data=build_data, test_train='test')
  # X_subjects, y_subjects = load_text_data(build_data=build_data)


  # X_data = np.array(X_data)
  # y_data = np.array(y_data)
  # X_data = decomposition_pca_train(X_data)
  # print "PCA Finished"


  # normalize_data(X_data)
  # normalize_data(test_data)

  # clfJoined = svm.SVC()
  # all_data = np.hstack((X_data, X_subjects))
  # all_test_data =np.hstack((test_data, test_subjects))
  # clfJoined.fit(all_data, y_data)
  # print "SHOWING COMBINED FEATURES SCORE"
  # show_score(clf, all_test_data, test_labels)

  # ## Let's see what is correlated in hurr!
  # cor = np.corrcoef(X_data, rowvar=0)
  # for i in range(0,len(X_data[0])-2):
  #   for j in range(i+1, len(X_data[0])-1):
  #     c = cor[i,j]
  #     if c > .65 or c < -.65:
  #       print "Kinda High Correlation for (%i, %i): %f" % (i, j, c)


  # # 


  # ## NOW LET"S TRY AND GET DOWN ON SOME CROSS VALIDATION
  # clf= svm.SVC()
  # # scores = cv.cross_val_score(clf, X_data, y_data, cv=5, scoring='roc_auc')
  # # print "AUC SCORES FROM CV"
  # # print scores  
  # # print "SCORES FROM TEST DATA"
  # clf.fit(X_data, y_data)
  # f=open(model_directory + 'svmBasicMinusEvalsGeneric', 'w')
  # pickle.dump(clf, f)
  # f.close()
  # show_score(clf, test_data, test_labels)



  
  ### THIS IS THE ORIGINAL - BESAST MODE!
  # X_train, X_test, y_train, y_test = split_data(X_data, y_data)
  # print "Data Split"
  # print len(X_train[0])
  # print X_train[0]

  # print len(test_data[0])
  # print test_data[0]


  # if 1:
  #   clf= svm.SVC()
  #   clf.fit(X_train, y_train)
  #   f=open(model_directory + 'svmBasic', 'w')
  #   pickle.dump(clf, f)
  #   f.close()
  # else:
  #   f=open(model_directory + 'svmBasic', 'r')
  #   clf = pickle.load(f)
  #   f.close()
  # # # logReg = 
  # print "Model Trained, Here is how it is looking on validation data:"
  # show_score(clf, X_test, y_test)

  # print "Model Trained, Here is how it is looking on Test Data:"
  # show_score(clf, test_data, test_labels)  
