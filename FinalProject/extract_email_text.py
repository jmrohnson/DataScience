import pg
from dbs import helpspotDB



query = """
select 
  a.request_id,
  messages
from helpspot_request a
 LEFT OUTER JOIN
   (
   select
       xrequest,
       smoosh(case when tnote ~ '<\\S*>' then regexp_replace(tNote, '<p>|</p>|<div>|</div>|<br \>', '', 'g') else tnote end, '</br><b>***END MESSAGE***</br></br>***BEGIN MESSAGE***</b></br>')  as messages
       from hs_request_history
       where tnote != ''
         and finitial = 1
       group by xrequest
   ) foo on (xrequest = a.request_id)
where date_month(a.open_timestamp) between '2013-05' and '2013-09'
  and request_id in (760315)
limit 5
"""

#, 760705, 760918, 761178, 761426, 762754, 762856, 762953, 763574, 764263, 764333, 765570, 767214, 768042, 768629, 768818, 769219, 769266, 769585, 769837


if __name__ == "__main__":
  hsdb = pg.connect(helpspotDB['db_name'], helpspotDB['db_server'], helpspotDB['db_port'], None, None, helpspotDB['db_username'], helpspotDB['db_password'])

  emails = hsdb.query(query).dictresult()

  print emails[0]
  hsdb.close()
