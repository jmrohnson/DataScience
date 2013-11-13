import matplotlib.pyplot as plt
import time
import string
import operator
from read_csv import read_csv
# from read_csv import read_csv_to_numpy_array

data_directory = '/Users/rjohnson/Documents/DS/DataScience/FinalProject/data/'
# Get TRAIN the data
request_data = read_csv(data_directory + 'request_info_train.txt')
# Get Labels
labels = [request[11] for request in request_data]

#Get Subjects for Text Analysis
subjects = [request[9] for request in request_data]

start = time.time()

# def bigrams(words):
#   wprev = None
#     for w in words:
#         yield (wprev, w)
#         wprev = w
#   yield (wprev, None)

bigrams = True

huck = {}
for subject in subjects:
    line = subject.split()
    if not bigrams:
      for word in line:
          word = word.lower()
          new_word = word.translate(string.maketrans("",""), string.punctuation)
          if new_word in huck:
              huck[new_word] += 1
          else:
              huck[new_word] = 1
    else:
      wprev = None
      for word in line:
        new_word = word.translate(string.maketrans("",""), string.punctuation)
        pair = (wprev, new_word)
        if pair in huck:
            huck[pair] += 1
        else:
            huck[pair] = 1
        wprev = new_word
      pair = (wprev, None)
      if pair in huck:
        huck[pair] += 1
      else:
        huck[pair] = 1


sorted_huck = sorted(huck.iteritems(), key=operator.itemgetter(1), reverse = True)
elapsed = time.time() - start

print 'Run took ', elapsed, ' seconds.'
print 'Number of distinct words: ', len(sorted_huck)

# Printing and plotting most popular words
npopular = 100
x = range(npopular)
y = []
for pair in range(npopular):
    y = y + [sorted_huck[pair][1]]
    print sorted_huck[pair]

plt.plot(x, y, 'ro')
plt.xlabel('Word ranking')
plt.ylabel('Word frequency')
plt.show()