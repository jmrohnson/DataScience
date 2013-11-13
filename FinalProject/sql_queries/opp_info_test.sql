select
  a.request_id
  ,date_month(open_timestamp) as month_partition
  ,count(e.*) as total_opps_btf
  ,sum(case when lower(e.product) = 'jira' then 1 else 0 end) as jira_opp_email
  ,sum(case when lower(e.product) = 'confluence' then 1 else 0 end) as conf_opp_email
  ,sum(case when lower(e.product) = 'bamboo' then 1 else 0 end) as bamboo_opp_email
  ,sum(case when lower(e.product) = 'confluence team calendars' then 1 else 0 end) as team_cal_opp_email
  ,sum(case when lower(e.product) = 'crowd' then 1 else 0 end) as crowd_opp_email
  ,sum(case when lower(e.product) = 'crucible' then 1 else 0 end) as crucible_opp_email
  ,sum(case when lower(e.product) = 'fisheye' then 1 else 0 end) as fisheye_opp_email
  ,sum(case when lower(e.product) = 'jira agile' then 1 else 0 end) as jira_agile_opp_email
  ,sum(case when lower(e.product) = 'jira capture' then 1 else 0 end) as jira_capture_opp_email
  ,sum(case when lower(e.product) = 'stash capture' then 1 else 0 end) as stash_opp_email
from helpspot_request a left outer join  evaluation_opportunity e
on open_timestamp > created_date and (end_date is null or open_timestamp < end_date)
  and email = customer_email
where date_month(a.open_timestamp) = '2013-10'
  and case_level is not null
  and a.subject != 'Delivery Status Notification (Failure)'
group by a.request_id, month_partition