select
   a.request_id
  ,date_month(open_timestamp) as month_partition
  ,a.open_timestamp
  ,date_trunc('day', a.open_timestamp) as day
  ,date_day(a.open_timestamp) as day_with_name
  ,date_trunc('week', a.open_timestamp) as week
  ,a.customer_email
  ,a.customer_email_domain
  ,case when a.customer_email_domain like '%@%' then 1 else 0 end as gmail_like_domain  ,trim(both ' ' from a.subject)
  ,a.region
  ,substring(a.case_level, 'Ops Level ([1-4])')
from helpspot_request a
where date_month(a.open_timestamp) = '2013-10'
  and a.subject != 'Delivery Status Notification (Failure)'
  and case_level is not null
order by a.request_id;