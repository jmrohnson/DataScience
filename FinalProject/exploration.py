import sys
import time
import pickle
import matplotlib as plt

from dtypes import dtypes, ind, day_mapping, region_mapping

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn import svm
from sklearn import decomposition
from sklearn import cross_validation as cv
from sklearn import grid_search as gs
from sklearn import metrics


from sklearn.cluster import KMeans, MiniBatchKMeans

import numpy as np




from read_csv import read_csv, read_csv_to_dict
# from read_csv import read_csv_to_numpy_array

data_directory = '/Users/rjohnson/Documents/DS/DataScience/FinalProject/data/'




def subjectKMeans(subjects, labels):
    #Set up Text Vecor (many options here)
  true_k = np.unique(labels).shape[0]
  vectorizer = TfidfVectorizer(max_df=0.5, max_features=1000,
                                  stop_words='english', use_idf=True)
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

# base should be vector of vectors, addenda should be a dict with keys equal to the first values in base
def append_data(base, addenda):
  for row in base:
    if addenda[row[0]]:
      row += addenda[row[0]]


offset = 1 # number of position changes things will go through, global var

def get_labels(base, func):
  labels = [func(request[ind['case_level']]) for request in base]
  return labels

def remove_data(data, indices):
  if not indices:
    return data
  else:
    to_remove = indices.pop()
    if len(data[0]) == to_remove + 1:
      new_data = [row[:to_remove] for row in data]
    else:
      new_data = [np.concatenate((row[:to_remove], row[to_remove+1:])) for row in data]
    return remove_data(new_data, indices)

def select_data(np_array, col_indices):
  return np_array[:,col_indices]

def convert_region_to_int(np_array, col):
  if col == 'region':
    mapping = region_mapping
  elif col == 'day_name':
    mapping = day_mapping
  for index in col_indices:
    types = np.unique(select_data(np_array, [col]))
    i = 0
    mapping = {}
    for t in types:
      mapping[t] = i
      i+=1
    j = 0
    for row in np_array:
      row[index] = mapping[row[col]]

      #Check out vectorizer for this
      #TF IDF Vectorizer
      
    
def load_data():

 # Get TRAIN the data, this is the main data with the label
  request_data = read_csv(data_directory + 'request_info_train.txt')
  # Append BTF sale
  btf_info = read_csv_to_dict(data_directory + 'btf_info_train.txt')
  append_data(request_data, btf_info)
  # Append OD sale
  od_info = read_csv_to_dict(data_directory + 'od_info_train.txt')
  append_data(request_data, od_info)
  # Append sale info
  sale_info = read_csv_to_dict(data_directory + 'sale_info_train.txt')
  append_data(request_data, sale_info)
  # Append Opportunity INFO
  opp_info = read_csv_to_dict(data_directory + 'opp_info_train.txt')
  append_data(request_data, opp_info)
  # Get Labels
  labels = get_labels(request_data, lambda x: x == 1)
  
  
  #Get Subjects for Text Analysis
  subjects = [request[ind['subject']] for request in request_data]

  #UnCorrelate JIRA and Agile
  # request_data = [row + 1]

  request_data = [tuple(row) for row in request_data]
  print request_data[0]
  np_request_data = np.array(request_data)

  print np_request_data[0]
  convert_category_to_int(np_request_data, 'region')
  convert_category_to_int(np_request_data, 'day_name')
  print np_request_data[0]

  np_request_data = np_request_data.astype('i8')

  only_features = remove_data(np_request_data, 
    sorted([ind['month'],
           ind['id'],
           ind['opened'],
           ind['day'],
           ind['week'],
           ind['email'],
           ind['email_domain'],
           ind['subject']]) )
  print only_features[0]

  # f = open('features', 'w')

  # pickle.dump(only_features, f)

  # f.close()

  return only_features, labels


def plot_distribution(data, index):
  col = data[:][0]
  plt.hist(col)
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
    print "Gonna start fitting something"
    start = time.time()
    clf= svm.SVC()
    clf.fit(features, result)
    print "Fit took this long:"
    print time.time() - start
    return clf

def predict(clf, features):
    """ Predict labels from trained CLF """
    return clf.predict(features).astype(np.int)

def show_score(clf, X_test, y_test):
    """ Scores are computed on the test set """
    y_pred = predict(clf, X_test)
    print metrics.classification_report(y_test.astype(np.int), y_pred)

def write_data(filename, data):
    """ Write numpy array into CSV """
    np.savetxt(filename, data, fmt='%d')



if __name__ == "__main__":
  # request_data = read_csv_to_numpy_array(data_directory + 'request_info_train.txt')

    X_data, y_data = load_data()
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    X_data = decomposition_pca_train(X_data)
    print "PCA Finished"
    print len(X_data)
    print len(y_data)

    X_train, X_test, y_train, y_test = split_data(X_data, y_data)
    print "Data Split"
    # clf = train(X_train, y_train)
    # clf= svm.SVC()
    # logReg = 
    print "Model Trained"
    # show_score(clf, X_test, y_test)

  # btf_info = read_csv(data_directory + 'request_info_train.txt', 9)
  # od_info = read_csv(data_directory + 'request_info_train.txt', 10)
  # # recent_request_history = read_csv(data_directory + 'recent_request_history_train.txt', 12)
  # opp_info = read_csv(data_directory + 'request_info_train.txt', 13)
  # sale_info = read_csv(data_directory + 'recent_request_history_train.txt', 10)
