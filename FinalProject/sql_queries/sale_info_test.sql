select
   a.request_id
   ,sum(case when date < open_timestamp and customer_email = email then 1 else 0 end) as prev_sales_email
   ,sum(case when date < open_timestamp and customer_email = email and date between open_timestamp - interval '1 month' and open_timestamp then 1 else 0 end) as prev_sales_email_month
   ,round(sum(case when date < open_timestamp and customer_email = email then amount else 0 end)) as prev_sales_dollars_email
   ,round(sum(case when date < open_timestamp and customer_email = email and date between open_timestamp - interval '1 month' and open_timestamp then amount else 0 end)) as prev_sales_dollars_email_month
   ,sum(case when date < open_timestamp and customer_email_domain = email_domain then 1 else 0 end) as prev_sales_email_domain
   ,sum(case when date < open_timestamp and customer_email_domain = email_domain and date between open_timestamp - interval '1 month' and open_timestamp then 1 else 0 end) as prev_sales_email_domain_month
   ,round(sum(case when date < open_timestamp and customer_email_domain = email_domain then amount else 0 end)) as prev_sales_dollars_email_domain
   ,round(sum(case when date < open_timestamp and customer_email_domain = email_domain and date between open_timestamp - interval '1 month' and open_timestamp then amount else 0 end)) as prev_sales_dollars_email_domain_month
from helpspot_request a  left outer join sale s
  on s.email_domain = customer_email_domain
    and date < open_timestamp 
where date_month(a.open_timestamp) = '2013-09'
  and case_level != ''
  and a.subject != 'Delivery Status Notification (Failure)'
group by a.request_id