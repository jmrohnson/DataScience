select
  contracttime, contracttype, count(*)
from hw3
group by 1, 2;

select 
  contracttime, contracttype, avg(salarynormalized), stddev(salarynormalized), count(*)
from hw3
where salarynormalized < 50000
group by 1, 2;

select locationnormalized, avg(salarynormalized), stddev(salarynormalized), count(*) from hw3 group by 1 having count(*) > 50 order by 2 desc

;

select 
case when title ilike '%engineer%' then 'eng'
     when title ilike '%nurse%' then 'nurse'
     when title ilike '%manager%' then 'manager'
     when title ilike '%exec%' then 'exec'
     when title ilike '%analyst%' then 'analyst'
     when title ilike '%develop%' then 'dev'
     when title ilike '%teach%' then 'teacher'
     when title ilike '%chef%' then 'chef'
 else 'other' end as titletype
, count(*), avg(salarynormalized), stddev(salarynormalized) 
from hw3 
  where salarynormalized < 50000
group by 1 order by 2 desc;


select 
  category
, count(*), avg(salarynormalized), stddev(salarynormalized) 
from hw3 
  where salarynormalized < 50000
group by 1 order by 2 desc;
;
select
  case when len < 501 then 'low'
       when len between 501 and 1000 then 'lower'
       when len between 1001 and 1500 then 'mid'
       when len between 1501 and 2000 then 'longer'
       when len > 2000 then 'long'
  end as lenCat
  , count(*), avg(salarynormalized), stddev(salarynormalized)
   from 
  (select char_length(fulldescription) as len, * from hw3) as a
where salarynormalized < 50000
group by 1 order by 1 asc;



select * from 
(
  select 
case when title ilike '%engineer%' then 'eng'
     when title ilike '%nurse%' then 'nurse'
     when title ilike '%manager%' then 'manager'
     when title ilike '%exec%' then 'exec'
     when title ilike '%analyst%' then 'analyst'
     when title ilike '%develop%' then 'dev'
     when title ilike '%teacher%' then 'teacher'
 else 'other' end as titletype, *
 from hw3) a
 where titletype = 'other'
  and salarynormalized < 50000;
  
  
select 
  company
  , count(*), avg(salarynormalized), stddev(salarynormalized) 
from hw3 
  where salarynormalized < 50000
group by 1 having count(*) > 20 order by 3 desc