import pg
import re
import read_csv
import pickle
import csv
from dbs import helpspotDB
  

query = """
select 
  a.request_id,
  message
from helpspot_request a
 LEFT OUTER JOIN
   (
   select
       xrequest,
       tnote  as message
       from hs_request_history
       where tnote != ''
         and finitial = 1
   ) foo on (xrequest = a.request_id)
where date_month(a.open_timestamp) between '2013-05' and '2013-09'
  and request_id in %s
"""


output = []
practice_list = ['760705', '760918', '761178', '761426', '762754', '762856', '762953', '763574', '764263', '764333', '765570', '767214', '768042', '768629', '768818', '769219', '769266', '769585', '769837']


def collect_text(requests, connection):
  i = 0
  so_far = 0
  reqs_to_get = []
  length = len(requests) -1
  for j, req in enumerate(requests):
    req_id = str(req[0])
    reqs_to_get.append(req_id) 
  
    if i < 100 and j != length:  
      i+=1
    else:
      if j == length:
        print "Last One!"
      print "Sweet, gonna go get some request text"
      so_far += 100
      print "So far we've done: %i" % so_far
      run_query_and_save_result(connection, reqs_to_get)
      i = 0

def list_to_string(l):
  out = '(' + ','.join(l)+')'
  return out

def run_query_and_save_result(connection, request_list):
  global output
  query_to_run = query % list_to_string(request_list)
  reqs = connection.query(query_to_run).getresult()
  for req in reqs:
    req_id = req[0]
    if req[1]:
      message = re.sub('(<[^>]*>)|(\n|\r)', ' ', req[1]).strip()
    else:
     message = ' '
    output.append([req_id, message])


if __name__ == "__main__":

  data_directory = '/Users/rjohnson/Documents/DS/DataScience/FinalProject/data/'
  requests = read_csv.read_csv(data_directory + 'request_info_train.txt')
  num_requests = len(requests)
  print "Number of Request = %i " % len(requests)
  hsdb = pg.connect(helpspotDB['db_name'], helpspotDB['db_server'], helpspotDB['db_port'], None, None, helpspotDB['db_username'], helpspotDB['db_password'])
  for i in range(0, num_requests/1000):
    collect_text(requests[i*1000:min((i+1)*1000 -1, num_requests)] , hsdb)
    print "Savint to CSV %i" % i
    out = csv.writer(open("email_text_tmp_%i.txt" %i,"w"), delimiter='|')
    for row in output:
      out.writerow(row)
    output=[]

  #run_query_and_save_result(hsdb, practice_list)

  
  hsdb.close()
