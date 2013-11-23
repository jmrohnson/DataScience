import numpy as np


def as_int(i):
  try:
    return int(i)
  except Exception:
    pass
  return i


def read_csv(file_name):
  read_file = open(file_name, 'r')
  to_return = []
  i = 0
  num_fields = 0
  for line in read_file:
    row = line.split('|')
    row = map(lambda s: as_int(s.strip()), row)
    if i == 0:
      num_fields = len(row)
    elif len(row) == num_fields:
      to_return.append(row)
    i+=1
  return to_return

def read_email_file(file_name):
  read_file = open(file_name, 'r')
  to_return = {}
  for line in read_file:
    row = line.split('|')
    if not row[0] in to_return:  
      row = map(lambda s: as_int(s.strip()), row)
      key = row[0]
      if len(row) > 2:
        email = ""
        for i in row[1:]: 
          email += " " + str(i)
      else:
        email = row[1]
      to_return[key] = email
  return to_return


def read_csv_to_dict(file_name):
  read_file = open(file_name, 'r')
  to_return = {}
  i = 0
  num_fields = 0
  for line in read_file:
    row = line.split('|')
    row = map(lambda s: as_int(s.strip()), row)
    if i == 0:
      num_fields = len(row)
    elif len(row) == num_fields:
      key = row[0]
      to_return[key] = row[1:]
    i+=1

  return to_return

def read_csv_to_numpy_array(file_name):
  with open(file_name, 'r') as read_file:

    lines = sum(1 for line in read_file)
    data = []
    i = 0 
    headers = []
    num_fields = 0
    for line in read_file:
      row = line.split('|')
      row = map(lambda s: s.strip(), row)
      print i
      if i == 0:
        print "Found Headers"
        print row
        num_fields = len(row)
        headers = row

      else:
        if len(row) != num_fields:
          to_return.append(row)
      i+=1 
    data_types = [(header, str) for header in headers]
    print data_types
    to_return = np.array(data, dtype = data_types)
    return to_return

