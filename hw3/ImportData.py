import pg
import sys
sys.path.insert(0, '/Users/rjohnson/.creds/')

from credentials import local

test = pg.DB(local['db_name'], local['db_server'], local['db_port'], None, None, local['username'], local['password'])

toRead =open('trainImport.csv', 'r')

i = 0
right = 0
big = 0
small = 0
out = []
for line in toRead:
  row= []
  add = True
  temp = ''
  raw = line.strip().split(',')
  for item in raw:
    item = item.strip()
    if add:
      if item and item[0] == '"':
        add = False
        temp += item
      else:
        row.append(item)
    else:
      if item and item[len(item)-1] == '"':
        temp += item
        row.append(temp)
        temp = ''
        add = True
      else:
        temp += item
    if len(row) == 12:
      right += 1
      try:
        test.inserttable('hw3', [row])
      except MemoryError, e:
        pass
      out.append(row)
    if len(row) > 12:
      big += 1
    elif len(row) < 12:
      small += 1
# for i in range(0,999):      
#   test.inserttable('hw3', out[i*10:(i+1)*10])
test.close()

