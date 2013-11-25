import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

import itertools
import numpy as numpy
from operator import itemgetter

import load_data

import logging
logging.basicConfig(filename='wordCount.log',level=logging.DEBUG)

def word_type(wordArray, wordType='single',):
  if wordType == 'bigram':
    for line in wordArray:
      wprev = None
      bigramArray= []
      for word in line:
        pair = (wprev, word)
        bigramArray.append(pair)
        wprev = word
      pair = (wprev, None)
      bigramArray.append(pair)
    return bigramArray
  elif wordType == 'trigram':
    for line in wordArray:
      wprev = None
      wwprev = None
      trigramArray= []
      for word in line:
        group = (wwprev, wprev, word)
        trigramArray.append(group)
        wwprev = wprev
        wprev = word
      group = (wwprev, wprev, None)
      trigramArray.append(group)
    return trigramArray
  else:
    return wordArray
  
def wordDiff(features, labels, wordType):
  posWords = {}
  negWords = {}
  pos_total = sum([1 if l == 1 else 0 for l in labels])
  neg_total = len(labels) - pos_total
  for i, words in enumerate(features):
    words = word_type(words, wordType=wordType)
    if labels[i]== 1:
      for w in words:
        if w in posWords:
          posWords[w] +=1
        else:
          posWords[w] =1
    else:
      for w in words:
        if w in negWords:
          negWords[w] +=1
        else:
          negWords[w] =1
  posPercs = []
  negPercs = []
  allWords = {}
  onlyPos = []
  onlyNeg = []
  for word in posWords:
    perc = 1.*posWords[word]/pos_total
    posPercs.append((word, perc))
    if word in negWords:
      allWords[word] = perc
    else:
     onlyPos.append((word, posWords[word], perc))
  for word in negWords:
    perc = 1.*negWords[word]/neg_total
    negPercs.append((word, perc))
    if word in allWords:
      allWords[word] -= perc
    else:
      onlyNeg.append((word, negWords[word], perc))
  combinedWords = []
  for word in allWords:
    combinedWords.append((word, perc))
  return {'combined':combinedWords, 'pos': onlyPos, 'neg': onlyNeg}

def analyzeWordDiff(d):
  posBig = sorted(d['pos'],key=itemgetter(1), reverse=True)
  negBig = sorted(d['neg'],key=itemgetter(1), reverse=True)
  comboPos = sorted(d['combined'],key=itemgetter(1), reverse=True)
  comboNeg = sorted(d['combined'],key=itemgetter(1),)
  logging.info("====================================================================================")
  logging.info("Most Postive of Common Words")
  logging.info(comboPos[:100])
  logging.info("====================================================================================")
  logging.info("Most Negative of Common Words")
  logging.info(comboNeg[:100])
  logging.info("====================================================================================")
  logging.info("Most of the ONLY Positive!")
  logging.info(posBig[:100])
  logging.info("====================================================================================")
  logging.info("Most of the ONLY Negative")
  logging.info(negBig[:100])
  


def evaluate_classifier(featx, train, test, train_labels, test_labels):
  trainfeats = [(best_bigram_word_feats(train[i].split(' ')), train_labels[i]) for i in range(len(train_labels))]
  testfeats = [(best_bigram_word_feats(test[i].split(' ')), test_labels[i]) for i in range(len(test_labels))]
  classifier = NaiveBayesClassifier.train(trainfeats)
  refsets = collections.defaultdict(set)
  testsets = collections.defaultdict(set)
  for i, (feats, label) in enumerate(testfeats):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
  print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
  print 'pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos'])
  print 'pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos'])
  print 'neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg'])
  print 'neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg'])
  classifier.show_most_informative_features()

def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=400):
  bigram_finder = BigramCollocationFinder.from_words(words)
  bigrams = bigram_finder.nbest(score_fn, n)
  return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

if __name__ == "__main__":
  train_subject, test_subject = load_data.load_email_data(0,  extension='unVectorized', stemmer='PorterStemmer', vectorizer='unVectorized')
  train_email, test_email = load_data.load_email_data(0,  extension='unVectorized', stemmer='PorterStemmer', vectorizer='unVectorized')
  train_features, train_labels = load_data.load_feature_data(0, test_train='train')
  test_features, test_labels = load_data.load_feature_data(0, test_train='test')
  train_email_features = [email.split(' ') for email in train_email]
  test_email_features = [email.split(' ') for email in test_email]
  train_subject_features = [subject.split(' ') for subject in train_subject]
  test_subject_features = [subject.split(' ') for subject in test_subject]
  for email_group in ['email', 'subject']:
    for wordType in ['single', 'bigram', 'trigram']:
      logging.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      logging.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      logging.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      logging.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      logging.info("Getting Word Stuff for %s and %s" % (email_group, wordType))
      if email_group == 'email':
        analyzeWordDiff(wordDiff(train_email_features, train_labels, wordType))
      else:
        analyzeWordDiff(wordDiff(train_subject_features, train_labels, wordType))

    

