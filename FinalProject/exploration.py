import sys
import time
import pickle
import math
import matplotlib.pyplot as plt

from dtypes import dtypes, ind, day_mapping, region_mapping

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn import svm
from sklearn import decomposition
from sklearn import cross_validation as cv
from sklearn import grid_search as gs
from sklearn import metrics


from sklearn.cluster import KMeans, MiniBatchKMeans

import numpy as np
import sklearn.preprocessing as preprocessing



from read_csv import read_csv, read_csv_to_dict
# from read_csv import read_csv_to_numpy_array

data_directory = '/Users/rjohnson/Documents/DS/DataScience/FinalProject/data/'
model_directory = '/Users/rjohnson/Documents/DS/DataScience/FinalProject/models/'

vars_to_remove = [
             ind['month'],
             ind['id'],
             ind['opened'],
             ind['day'],
             ind['week'],
             ind['email'],
             ind['email_domain'],
             ind['subject'],
             ind['total_od_licenses'],
             ind['total_od_evals'],
             ind['case_level']]

offset = len(vars_to_remove)

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

def subjectTextVectorizer(request_info):
  subjects = select_data(request_info, ind['subject'])
  # nltkSTep
  # stemmingStep
  vectorizer = TfidfVectorizer(stop_words='english', use_idf=True,
                                  norm='l1', 
                                  ngram_range=(1, 3),
                                  lowercase=True, strip_accents='unicode'
                                  )
  subject_features = vectorizer.fit_transform(subjects)
  print subject_features[0]
  return subject_features



# base should be vector of vectors, addenda should be a dict with keys equal to the first values in base
def append_data(base, addenda):
  for row in base:
    if addenda[row[0]]:
      row += addenda[row[0]]

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

def convert_category_to_int(np_array, col):
  if col == 'region':
    mapping = region_mapping
  elif col == 'day_name':
    mapping = day_mapping
  index = ind[col]
  types = np.unique(select_data(np_array, index))
  for row in np_array:
    row[index] = mapping[row[index]]

      #Check out vectorizer for this
      #TF IDF Vectorizer
      
def segment_of_level_ones(train_set, train_labels):
  output = []
  quarter = len(train_set)/4
  for i, r enumerate(train_set):
    if train_labels[i] == 1:


    
def load_feature_data(build_data=0, test_train='train'):
  if build_data in (1, 3) :
    print "Building Data Fresh"
    # Get TRAIN the data, this is the main data with the label
    request_data = read_csv(data_directory + 'request_info_' + test_train + '.txt')
    # Append BTF sale
    btf_info = read_csv_to_dict(data_directory + 'btf_info_' + test_train + '.txt')
    append_data(request_data, btf_info)
    # Append OD sale
    od_info = read_csv_to_dict(data_directory + 'od_info_' + test_train + '.txt')
    append_data(request_data, od_info)
    # Append Opportunity INFO
    opp_info = read_csv_to_dict(data_directory + 'opp_info_' + test_train + '.txt')
    append_data(request_data, opp_info)
    # Append sale info
    sale_info = read_csv_to_dict(data_directory + 'sale_info_' + test_train + '.txt')
    append_data(request_data, sale_info)
    # Get Labels
    # labels = get_labels(request_data, lambda x: x if x < 3 else 3)
    labels = get_labels(request_data, lambda x: x == 1)

    #UnCorrelate JIRA and Agile
    # request_data = [row + 1]

    request_data = [tuple(row) for row in request_data]
    np_request_data = np.array(request_data)

    convert_category_to_int(np_request_data, 'region')
    convert_category_to_int(np_request_data, 'day_name')




    only_features = remove_data(np_request_data, 
      sorted(vars_to_remove) )

    only_features = np.array(only_features).astype('i8')

    #Bring the Dollar Amounts down to Log Scale
    dollar_features = ['prev_sales_dollars_email', 'prev_sales_dollars_email_month', 
                       'prev_sales_dollars_email_domain', 'prev_sales_dollars_email_domain_month']
    
    for feature in dollar_features:
      index = ind[feature] - offset  # Minus 8 because we removed 8 features above
      for row in only_features:
        amount = row[index]
        if amount < 0:
          row[index] = (-1)*math.log(float((-1)*amount))
        elif amount > 0:
          row[index] = math.log(float(amount))
    

    print "Data Built Fresh. It looks like this:"
    print only_features[0]

    f = open('features_'+test_train+ '.pkl', 'w')
    pickle.dump(only_features, f)
    f.close()
    f = open('labels_'+test_train+ '.pkl', 'w')
    pickle.dump(labels, f)
    f.close()
  else:
    print "Loading Basic Features Pickled Data"
    f = open('features_'+test_train+ '.pkl', 'r')
    only_features = np.array(pickle.load(f))
    f.close()
    f = open('labels_'+test_train+'.pkl', 'r')
    labels = np.array(pickle.load(f))
    f.close()
    print "Data Loaded from pickle. It looks like this:"
    print only_features[0]

  return only_features, labels


def load_text_data(build_data=0):
  if build_data in (2, 3):
    print "Loading Subject Text Data"
    if build_data ==2:
      train_data = read_csv(data_directory + 'request_info_train.txt')
      test_data = read_csv(data_directory + 'request_info_test.txt')

    all_data = train_data + test_data
    len_train = len(train_data)
    #Get Subjects for Text Analysis
    subject_features = subjectTextVectorizer(np.array(all_data))

    train_features = subject_features[:len_train]
    test_features = subject_features[len_train:]

    f = open('subject_features_train.pkl', 'w')
    pickle.dump(train_features, f)
    f.close()

    f = open('subject_features_test.pkl', 'w')
    pickle.dump(test_features, f)
    f.close()

  else:
    print "Pickling Subject Text Data"
    f = open('subject_features_traint', 'r')
    train_features = np.array(pickle.load(f))
    f.close()
    f = open('subject_features_test', 'r')
    test_features = np.array(pickle.load(f))
    f.close()
  print "Subject Text Data Returned"
  
  return train_features, test_features

def load_BIG_text_data(build_data=0):
  if build_data == 4:
    print "building Email Body Data"


def get_Big_email_text(filename):
  f = open(filename, 'r'):



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
  print "Gonna start fitting something"
  start = time.time()
  clf= svm.SVC()
  clf.fit(features, result)
  print "Fit took this long:"
  print time.time() - start
  return clf

def predict(clf, features):
  """ Predict labels from trained CLF """
  return clf.predict(features)

def show_score(clf, X_test, y_test):
  """ Scores are computed on the test set """
  y_pred = predict(clf, X_test)
  print metrics.classification_report(y_test, y_pred)

def write_data(filename, data):
  """ Write numpy array into CSV """
  np.savetxt(filename, data, fmt='%d')

def normalize_data(x_data):
  min_max_scaler = preprocessing.MinMaxScaler()
  category_data = select_data(x_data, [0,1])
  numerical_data = select_data(x_data, range(2,45-offset))
  x_data = np.hstack((category_data, numerical_data))


if __name__ == "__main__":
  # request_data = read_csv_to_numpy_array(data_directory + 'request_info_train.txt')

  if len(sys.argv) > 1:
    build_data = int(sys.argv[1])
  else:
    print "you didn't say whether or not to build the data so we'll pickle it"
    build_data=0
  

  # PLOTTING SOME DATA
  # for i, item in enumerate(X_train):
  # # plot_distribution(X_train, 0, "Day")


  X_data, y_data = load_feature_data(build_data=build_data, test_train='train')
  test_data, test_labels = load_feature_data(build_data=build_data, test_train='test')
  X_subjects, y_subjects = load_text_data(build_data=build_data)


  # X_data = np.array(X_data)
  # y_data = np.array(y_data)
  # X_data = decomposition_pca_train(X_data)
  # print "PCA Finished"


  # normalize_data(X_data)
  # normalize_data(test_data)

  clf = svm.SVC()
  clf.fit(X_subjects, y_data)
  print "SHOWING JUST SUBJECT SCORE"
  show_score(clf, y_subjects, test_labels)


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
