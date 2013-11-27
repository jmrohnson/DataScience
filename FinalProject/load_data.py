import pickle
import re

from time import time

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import LancasterStemmer
from nltk import RegexpStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np
import sklearn.feature_selection as fs

from dtypes import dtypes, ind, day_mapping, region_mapping
from read_csv import read_csv, read_csv_to_dict, read_email_file


###################################################################################
# General Variables
###################
data_directory = '/Users/rjohnson/Documents/DS/DataScience/FinalProject/data/'

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

# Text Analysis Variables
extra_stops = ['nbsp']
my_stops = stopwords.words('english') + extra_stops
regexpForKeepingToken = '(^[a-zA-Z]{3,100}$)|\?|^(sen-)|^(at-)'
###################################################################################

###################################################################################
# Functions for manipulating Data
##############
def append_data(base, addenda):
  # base should be vector of vectors 
  # addenda should be a dict with keys equal to the first values in base
  for row in base:
    try:
      row.append(addenda[row[0]])
    except Exception:
      pass
      

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

#####################################################################################

#####################################################################################
#Load all request data except for the text
#################
def load_feature_data(build_data=0, test_train='train'):
  if build_data in (1, 4) :
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
    labels = get_labels(request_data, lambda x: x == 1)

    request_data = [tuple(row) for row in request_data]
    np_request_data = np.array(request_data)

    convert_category_to_int(np_request_data, 'region')
    convert_category_to_int(np_request_data, 'day_name')

    only_features = remove_data(np_request_data, sorted(vars_to_remove))

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

    f = open('pickle_data/features_'+test_train+ '.pkl', 'w')
    pickle.dump(only_features, f)
    f.close()
    f = open('pickle_data/labels_'+test_train+ '.pkl', 'w')
    pickle.dump(labels, f)
    f.close()
  else:
    print "Loading Basic Features Pickled Data"
    f = open('pickle_data/features_'+test_train+ '.pkl', 'r')
    only_features = np.array(pickle.load(f))
    f.close()
    f = open('pickle_data/labels_'+test_train+'.pkl', 'r')
    labels = np.array(pickle.load(f))
    f.close()
    print "Data Loaded from pickle. It looks like this:"
    print only_features[0]

  return only_features, labels

#####################################################################################


#####################################################################################
# Functions for manipulating Text Data
################

def remove_stopwords(list_of_words):
  to_remove = []
  for w in list_of_words:
    if w in my_stops or not re.match(regexpForKeepingToken, w):
      to_remove.append(w)
  list_of_words[:] = [w for i,w in enumerate(list_of_words) if w not in to_remove]

def stemming(word_list, stemmer_type=""):
    """
    -------------------------
    Returns stemmed words
    -------------------------

    Thanks Pradeep for this!
    
    """
    stemmed_words = []
    stemmer = None
    if stemmer_type == 'PorterStemmer':
        stemmer=PorterStemmer()
    elif stemmer_type == 'LancasterStemmer':
        stemmer = LancasterStemmer()
    elif stemmer_type == 'RegexpStemmer':
        stemmer = RegexpStemmer('ing$|s$|e$', min=3)

    for word in word_list:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words  

def TextVectorizer(text, vectorizer='TfidfVectorizer', num_features = None):
  if vectorizer == 'TfidfVectorizer':
    print "Building TfidfVectorizer Vector"
    vectorizer = TfidfVectorizer(use_idf=True, max_features=num_features,
                                 decode_error=u'ignore',
                                 ngram_range=(1, 3),
                                 lowercase=True, strip_accents='unicode'
                                 )
    text_features = vectorizer.fit_transform(text)
  elif vectorizer == 'HashingVectorizer':
    print "Building HashingVectorizer Vector"
    if not num_features:
      num_features = 2 ** 20
    vectorizer = HashingVectorizer(decode_error=u'ignore',
                                   ngram_range=(1, 3), n_features=num_features,
                                   lowercase=True, strip_accents='unicode'
                                   )
    text_features = vectorizer.fit_transform(text)
  return text_features

def tokenizeText(text, stemmer='PorterStemmer'):
  text = re.sub('\. ', ' ', text)
  text = re.sub('(at|AT)-[0-9]+', 'at-', text)
  text = re.sub('(sen|SEN|Sen)-[0-9]+', 'sen-', text)
  tokens = word_tokenize(text)
  tokens = [str(t).lower() for t in tokens]
  remove_stopwords(tokens)
  stemmed_words = stemming(tokens, stemmer)
  return ' '.join(stemmed_words)

def preProcessText (textArray, stemmer='PorterStemmer', vectorizer='TfidfVectorizer', num_features = None):
  print "Tokenizing"
  print vectorizer
  print len(textArray)
  newArray = []
  i = 0
  for line in textArray:
    if i % 10000 == 0:
      print "Tokenize this many lines so far: %i " % i
      print "Here is what the line looked like:"
      print line
    newline= tokenizeText(line, stemmer=stemmer)
    newArray.append(newline)
    if i % 10000 == 0:
      print "The Last line looked like:"
      print newline
    i+= 1
  print len(newArray)
  if vectorizer != 'unVectorized':
    tv =  TextVectorizer(newArray, vectorizer=vectorizer, num_features=num_features)
  else:
    return newArray


def buildEmailText(requests, rebuild = 1, stemmer='PorterStemmer', vectorizer='TfidfVectorizer', num_features = None):
  rawText = {}
  output = {}
  if rebuild:
    for i in range(0, 54):
      data_file = data_directory + 'email_text/email_text_tmp_test_' + str(i) + '.txt'
      print "Reading data from %s" % data_file
      emails = read_email_file(data_file)
      # for req_id in emails:
      #   text = emails[req_id]
      #   rawText{req_id}, text])
      rawText.update(emails)      
    f = open('pickle_data/rawEmailText.pkl', 'w')
    pickle.dump(rawText, f)
    f.close()
  else:
    f = open('pickle_data/rawEmailText.pkl', 'r')
    rawText = pickle.load(f)
    f.close()
  requests_only = select_data(requests, 0)
  requests_only = [[int(req_id)] for req_id in requests_only]
  append_data(requests_only, rawText)
  for row in requests_only:
    if len(row) < 2:
      row.append("")
  rawText = [row[1] for row in requests_only]
  print "New Raw Text Array %i" % len(rawText)
  print "Got data from files: "
  print len(rawText)
  return preProcessText(rawText, stemmer=stemmer, vectorizer=vectorizer, num_features=num_features)

  

def buildSubjectText(requests, stemmer='PorterStemmer', vectorizer='TfidfVectorizer', num_features = None):
  rawText = select_data(requests, ind['subject'])
  rawText = np.array(rawText)
  return preProcessText(rawText, stemmer=stemmer, vectorizer=vectorizer, num_features=num_features)

#####################################################################################

#####################################################################################
#Function to load text data
#################
def load_subject_data(build_data, extension, stemmer='PorterStemmer', vectorizer='TfidfVectorizer', num_features = None, 
                      labels=np.array([])):
  train_data = read_csv(data_directory + 'request_info_train.txt')
  test_data = read_csv(data_directory + 'request_info_test.txt')
  train_length = len(train_data)
  all_data = train_data + test_data
  
  if build_data in (2, 4):
    print "Loading Subject Text Data"   
    #Get Subjects for Text Analysis
    subject_features = buildSubjectText(np.array(all_data), stemmer=stemmer, vectorizer=vectorizer, num_features=num_features)
    f = open('pickle_data/subject_features_' + extension + '_' + stemmer + '.pkl', 'w')
    pickle.dump(subject_features, f)
    f.close()

  else:
    print "Pickling Subject Text Data"
    f = open('pickle_data/subject_features_' + extension + '_' + stemmer + '.pkl', 'r')
    subject_features = pickle.load(f)
    f.close()
    if labels.any():
      chi2 = fs.chi2(subject_features, labels)
      strong = []
      weak = []
      for i, p in enumerate(chi2[1]):
        if p < 0.1:
          weak.append(i)
          if p < .05:
            strong.append(i)
      strongFeatures = subject_features[:,strong]
      weakFeatures = subject_features[:,weak]
      f = open('pickle_data/subject_features_' + '_chi2_strong_' + extension + '_' + stemmer + '.pkl', 'w')
      pickle.dump(strongFeatures, f)
      f.close()
      f = open('pickle_data/subject_features_' + '_chi2_weak_' + extension + '_' + stemmer + '.pkl', 'w')
      pickle.dump(weakFeatures, f)
      f.close()
  
  train = subject_features[:train_length]
  test = subject_features[train_length:]
  return train, test


def load_email_data(build_data, extension, stemmer='PorterStemmer', vectorizer='TfidfVectorizer', num_features = None, 
                      labels=np.array([])):
  train_data = read_csv(data_directory + 'request_info_train.txt')
  test_data = read_csv(data_directory + 'request_info_test.txt')
  train_length = len(train_data)
  all_data = train_data + test_data

  if build_data in (3, 4):
    print "Loading Email Text Data"
    #Get Subjects for Text Analysis
    body_features = buildEmailText(np.array(all_data), stemmer=stemmer, vectorizer=vectorizer, num_features = num_features)
    f = open('pickle_data/body_features_' + extension + '_' + stemmer + '.pkl', 'w')
    pickle.dump(body_features, f)
    f.close()

  else:
    print "Pickling Email Text Features"
    f = open('pickle_data/body_features_' + extension + '_' + stemmer + '.pkl', 'r')
    body_features = pickle.load(f)
    f.close()
    if labels.any():
      chi2 = fs.chi2(body_features, labels)
      strong = []
      weak = []
      for i, p in enumerate(chi2[1]):
        if p < 0.1:
          weak.append(i)
          if p < .05:
            strong.append(i)
      strongFeatures = body_features[:,strong]
      weakFeatures = body_features[:,weak]
      f = open('pickle_data/body_features_' + '_chi2_strong_' + extension + '_' + stemmer + '.pkl', 'w')
      pickle.dump(strongFeatures, f)
      f.close()
      f = open('pickle_data/body_features_' + '_chi2_weak_' + extension + '_' + stemmer + '.pkl', 'w')
      pickle.dump(weakFeatures, f)
      f.close()

  print "Text Data Returned"

  train = body_features[:train_length]
  test = body_features[train_length:]
  return train, test



#####################################################################################

if __name__ == "__main__":
  train_features, train_labels = load_feature_data(0, test_train='train')
  test_features, test_labels = load_feature_data(0, test_train='test')
  all_labels = np.append(train_labels, test_labels)
  for stem in ['RegexpStemmer', 'PorterStemmer', 'LancasterStemmer']:
    for vectorizer in ['TfidfVectorizer']:
      print "Loading Data for %s, %s:" % (stem, vectorizer)
      test, train = load_subject_data(0, vectorizer, stemmer=stem, vectorizer=vectorizer, labels = all_labels)
      test, train = load_email_data(0, vectorizer, stemmer=stem, vectorizer=vectorizer, labels = all_labels)

  # Best Combinatino seemed to be regexpStemmer, HashingVectorizer
  # Now we play with num_features
  # stem = 'RegexpStemmer'
  # vectorizer = 'HashingVectorizer'
  # for n in [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]:
  #   print "Loading Data for %s, %s and max_featurs = %i" % (stem, vectorizer, n)    
  #   test_train = load_subject_data(2, str(n) + vectorizer, stemmer=stem, vectorizer=vectorizer, num_features=n)
  #   test_train = load_email_data(3, str(n) + vectorizer, stemmer=stem, vectorizer=vectorizer, num_features=n)



