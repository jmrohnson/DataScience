select
  a.request_id
  ,sum(case when open_timestamp between eval_date + interval '2 month' and expiry_date then 1 else 0 end) as total_od_licenses
  ,sum(case when open_timestamp between eval_date and eval_date + interval '2 month' then 1 else 0 end) as total_od_evals
  ,sum(case when lower(lo.base_product) = 'jira' and open_timestamp between eval_date + interval '2 month' and expiry_date then 1 else 0 end) as jira_od
  ,sum(case when lower(lo.base_product) = 'confluence' and open_timestamp between eval_date + interval '2 month' and expiry_date then 1 else 0 end) as conf_od
  ,sum(case when lower(lo.base_product) = 'jira agile' and open_timestamp between eval_date + interval '2 month' and expiry_date then 1 else 0 end) as jira_agile_od
  ,sum(case when lower(lo.base_product) = 'jira' and open_timestamp between eval_date and eval_date + interval '2 month' then 1 else 0 end) as jira_od_email_eval
  ,sum(case when lower(lo.base_product) = 'confluence' and open_timestamp between eval_date and eval_date + interval '2 month' then 1 else 0 end) as confluence_od_email_eval
  ,sum(case when lower(lo.base_product) = 'jira agile' and open_timestamp between eval_date and eval_date + interval '2 month' then 1 else 0 end) as jira_agile_od_email_eval
from helpspot_request a left outer join  license_ondemand lo
on open_timestamp > eval_date and (open_timestamp <expiry_date)
  and a.customer_email = lo.tech_email
where date_month(a.open_timestamp) between '2013-05' and '2013-09'
  and a.subject != 'Delivery Status Notification (Failure)'
  and case_level != ''
group by a.request_id