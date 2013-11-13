select 
  a.request_id,
  ,date_month(open_timestamp) as month_partition
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
  and case_level is not null
limit 5