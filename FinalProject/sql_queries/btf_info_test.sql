select
  a.request_id
  ,count(lo.*) as total_licenses_btf
  ,sum(case when lower(lo.base_product) = 'jira' then 1 else 0 end) as jira_btf_email
  ,sum(case when lower(lo.base_product) = 'confluence' then 1 else 0 end) as conf_btf_email
  ,sum(case when lower(lo.base_product) = 'jira agile' then 1 else 0 end) as jira_agile_btf_email
  ,sum(case when lower(lo.base_product) = 'jira' and open_timestamp between start_date and start_date + interval '1 month' then 1 else 0 end) as jira_btf_recent_email
  ,sum(case when lower(lo.base_product) = 'confluence' and open_timestamp between start_date and start_date + interval '1 month' then 1 else 0 end) as confluence_btf_recent_email
  ,sum(case when lower(lo.base_product) = 'jira agile' and open_timestamp between start_date and start_date + interval '1 month' then 1 else 0 end) as jira_agile_btf_recent_email
from helpspot_request a left outer join  (select * from license where not hosted) lo
on open_timestamp > start_date and (open_timestamp < expiry_date or expiry_date is null)
  and a.customer_email = lo.tech_email
where date_month(a.open_timestamp) = '2013-09'
  and case_level != ''
  and a.subject != 'Delivery Status Notification (Failure)'
group by a.request_id