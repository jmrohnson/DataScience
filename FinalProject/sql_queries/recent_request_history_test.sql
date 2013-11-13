select
  a.request_id
  ,date_month(a.open_timestamp) as month_partition
  ,sum(case when b.open_timestamp between a.open_timestamp - interval '1 year' and a.open_timestamp and a.customer_email = b.customer_email then 1 else 0 end) as prev_year_email
  ,sum(case when b.open_timestamp between a.open_timestamp - interval '6 months' and a.open_timestamp and a.customer_email = b.customer_email  then 1 else 0 end) as prev_6_months_email
  ,sum(case when b.open_timestamp between a.open_timestamp - interval '1 months' and a.open_timestamp and a.customer_email = b.customer_email  then 1 else 0 end) as prev_1_months_email
  ,sum(case when b.open_timestamp between a.open_timestamp - interval '1 week' and a.open_timestamp and a.customer_email = b.customer_email then 1 else 0 end) as prev_1_week_email
  ,sum(case when b.open_timestamp between a.open_timestamp - interval '1 year' and a.open_timestamp then 1 else 0 end) as prev_year_email_domain
  ,sum(case when b.open_timestamp between a.open_timestamp - interval '6 months' and a.open_timestamp then 1 else 0 end) as prev_6_months_email_domain
  ,sum(case when b.open_timestamp between a.open_timestamp - interval '1 months' and a.open_timestamp then 1 else 0 end) as prev_1_months_email_domain
  ,sum(case when b.open_timestamp between a.open_timestamp - interval '1 week' and a.open_timestamp then 1 else 0 end) as prev_1_week_email_domain
from helpspot_request a 
  left outer join (select customer_email, customer_email_domain, open_timestamp, case_level from helpspot_request where open_timestamp > '2012-05-01') b using(customer_email_domain)
where date_month(a.open_timestamp) = '2013-10'
  and a.case_level is not null
  and a.subject != 'Delivery Status Notification (Failure)'
group by a.request_id, month_partition