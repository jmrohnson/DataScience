dtypes = [
 ('id', '|S5'),
 ('month','|S5'),
 ('opened', '|S5'),
 ('day', '|S5'),
 ('day_name', '|S5'),
 ('week', '|S5'),
 ('email', '|S20'),
 ('email_domain', '|S20'),
 ('gmail_like', 'b'),
 ('subject', '|S100'),
 ('region', '|S10'),
 ('case_level', 'i4'),
  #Adding BTF License Info
 ('total_licenses', 'i4'),
 ('jira_btf', 'i4'),
 ('conf_btf', 'i4'),
 ('agile_btf', 'i4'),
 ('jira_btf_recent', 'i4'),
 ('conf_btf_recent', 'i4'),
 ('agile_btf_recent', 'i4'),
  #Addding OD Info
 ('total_od_licenses', 'i4'),
 ('total_od_evals', 'i4'),
 ('jira_od', 'i4'),
 ('conf_od', 'i4'),
 ('agile_od', 'i4'),
 ('jira_od_eval', 'i4'),
 ('conf_od_eval', 'i4'),
 ('agile_od_eval', 'i4'),
   #Adding BTF Opportunity Info
 ('total_opps', 'i4'), 
 ('jira_opp_email', 'i4'),
 ('conf_opp_email', 'i4'),
 ('bamboo_opp_email', 'i4'),
 ('team_cal_opp_email', 'i4'),
 ('crowd_opp_email', 'i4'),
 ('crucible_opp_email', 'i4'),
 ('fisheye_opp_email', 'i4'),
 ('jira_agile_opp_email', 'i4'),
 ('jira_capture_opp_email', 'i4'),
 ('stash_opp_email', 'i4'),
  #Adding Sales Info
 ('prev_sales_email', 'i4'),
 ('prev_sales_email_month', 'i4'),
 ('prev_sales_dollars_email', 'f8'),
 ('prev_sales_dollars_email_month', 'f8'),
 ('prev_sales_email_domain', 'i4'),
 ('prev_sales_email_domain_month', 'i4'),
 ('prev_sales_dollars_email_domain', 'f8'),
 ('prev_sales_dollars_email_domain_month', 'f8')
  ]

  # (736667, '2013-05', '2013-05-01 00:00:01', '2013-05-01 00:00:00', 'Wed', '2013-04-29 00:00:00', 
  #   'linda.cakau@oracle.com', 'oracle.com', 0, 'Atlassian Invoice AT-921873: Oracle', 
  #   'Americas', 1, 0, 0, 0, 0, 0, 0, 0,
  #    0, 0, 0, 0, 0, 0, 0, 0,
  #     1, 1, 1350, 1350, 425, 21, 790110, 11863,
  #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

day_mapping = {
  'Mon':1,
  'Tue':2,
  'Wed':3,
  'Thu':4,
  'Fri':5,
  'Sat':6,
  'Sun':7,
}


region_mapping = {
  'Americas':1,
  'APAC':2,
  'EMEA':3,
  '':4,
}

ind = {'id': 0,
  'month':1,
  'opened': 2,
  'day': 3,
  'day_name': 4,
  'week': 5,
  'email': 6,
  'email_domain': 7,
  'gmail_like': 8,
  'subject': 9,
  'region': 10,
  'case_level': 11,
  #Adding BTF License Info
  'total_licenses': 12,
  'jira_btf': 13,
  'conf_btf': 14,
  'agile_btf': 15,
  'jira_btf_recent': 16,
  'conf_btf_recent': 17,
  'agile_btf_recent': 18,
  #Addding OD Info
  'total_od_licenses': 19,
  'total_od_evals': 20,
  'jira_od': 21,
  'conf_od': 22,
  'agile_od': 23,
  'jira_od_eval': 24,
  'conf_od_eval': 25,
  'agile_od_eval': 26,
  #Adding BTF Opportunity Info
  'total_opps': 27,
  'jira_opp_email': 28,
  'conf_opp_email': 29,
  'bamboo_opp_email': 30,
  'team_cal_opp_email': 31,
  'crowd_opp_email': 32,
  'crucible_opp_email': 33,
  'fisheye_opp_email': 34,
  'jira_agile_opp_email': 35,
  'jira_capture_opp_email': 36,
  'stash_opp_email': 37,
  #Adding Sales Info
  'prev_sales_email': 38,
  'prev_sales_email_month': 39,
  'prev_sales_dollars_email': 40,
  'prev_sales_dollars_email_month': 41,
  'prev_sales_email_domain': 42,
  'prev_sales_email_domain_month': 43,
  'prev_sales_dollars_email_domain': 44,
  'prev_sales_dollars_email_domain_month': 45,
  }


rr = {
  'id': 'int',
  'month': 'str',
  'prev_year_email': 'int',
  'prev_6_months_email': 'int',
  'prev_1_months_email': 'int',
  'prev_1_week_email': 'int',
  'prev_year_email_domain': 'int',
  'prev_6_months_email_domain': 'int',
  'prev_1_months_email_domain': 'int',
  'prev_1_week_email_domain': 'int',
  }
