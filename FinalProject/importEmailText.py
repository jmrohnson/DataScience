
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer

from read_csv import read_email_file
import pickle
import re


data_directory = '/Users/rjohnson/Documents/DS/DataScience/FinalProject/data/'

extra_stops = ['nbsp']

my_stops = stopwords.words('english') + extra_stops
regexpForKeepingToken = '(^[a-z]{3,100})|\?'

def remove_stopwords(list_of_words):
  for w in list_of_words:
    if w in my_stops or not re.match(regexpForKeepingToken, w):
      list_of_words.remove(w)


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



if __name__ == "__main__":
  final_out = {}
  for i in range(0, 9):
    data_file = data_directory + 'email_text/email_text_tmp_test_' + str(i) + '.txt'
    print "Reading data from %s" % data_file
    emails = read_email_file(data_file)
    for req_id in emails:
      text = emails[req_id]
      tokens = word_tokenize(text)
      tokens = [str(t).lower() for t in tokens]
      remove_stopwords(tokens)
      stemmed_words = stemming(tokens, 'PorterStemmer')
      final_out[req_id] = ' '.join(stemmed_words)
      i+=1

  f = open('testEmailText.pkl', 'w')
  pickle.dump(final_out, f)
  f.close()


