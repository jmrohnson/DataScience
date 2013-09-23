allData = open('colleges.csv')
for line in allData:
  row=line.strip().split(',')
  # if len(row) > 1:  # We've hit the last line
    # print ','.join(item if item != '*' else 'NULL' for item in row )
  if len(row) > 1 and '*' not in row:
    print ','.join(item for item in row )







