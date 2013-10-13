##Define bucketing functions

getCityTypeSal <- function(cities){
  out <- vector()
  l <- length(cities)
  for (i in 1:l)
  {
    city <- cities[i]
    if (city %in% c('The City', 'London')){
      type = 'up'
    } 
    else if (city %in% c('South East London', 'Berkshire', 'Wales', 'Hampshire', 'Hertfordshire', 'West Midlands', 'Southampton', 'Cambridgeshire')) {
      type = 'upMid'
    }
    else  if (city %in% c('Surrey', 'Essex', 'Bristol', 'Belfast', 'Glasgow', 'Leicester', 'Reading', 'UK', 'Lancashire', 'Oxfordshire')) {
      type = 'lowMid'
    }
    else if(city %in% c('Nottingham', 'Bradford', 'Cheshire', 'Birmingham', 'Cambridge', 'Newcastle Upon Tyne', 'Liverpool', 'Manchester', 'Sheffield', 'West Yorkshire', 'Leeds')){
      type = 'low'   
    }
    else if (city == '')
    {
      type = 'none'
    }
    else{
      type = 'small'
    }
    out <- append(out, type)
  }
  return(out)
}

getSourceTypeSal <- function(sources){
  out <- vector()
  l <- length(sources)
  for (i in 1:l)
  {
    s <- sources[i]
    if (s %in% c('cwjobs.co.uk', 'rengineeringjobs.com', 'theitjobboard.co.uk', 'planetrecruit.com', 'Jobs24')){
      type = 'boss'
    } 
    else if (s %in% c('thecareerengineer.com', 'totaljobs.com', 'staffnurse.com', 'recruitni.com', 'jobsineducation.co.uk', 'hays.co.uk', 'strike-jobs.co.uk', 'fish4.co.uk', 'cv-library.co.uk', 'hotrecruit.com', 'careworx.co.uk')) {
      type = 'good'
    }
    else  if (s %in% c('Multilingualvacancies', 'MyUkJobs', 'nijobfinder.co.uk', 'caterer.com', 'jobs.catererandhotelkeeper.com', 'leisurejobs.com', 'Jobcentre Plus')) {
      type = 'weak'
    }
    else if (s == '')
    {
      type = 'none'
    }
    else{
      type = 'small'
    }
    out <- append(out, type)
  }
  return(out)
}

getCompanyType <- function(companies){
  out <- vector()
  l <- length(companies)
  for (i in 1:l)
  {
    company <- companies[i]
    if (company %in% c('Domus Recruitment', 'Chess Partnership', 'Matchtech Group plc.', 'Monarch Recruitment', 'MBN Recruitment', 'Albior Financial Recruitment', 'JOBG8', 'OCC Computer Personnel/RI', 'Remedy Recruitment Group Ltd', 'MatchBox Recruiting Ltd', 'Gregory Martin International', 'Support Services Group'))
    {
      type = 'top'
    } 
    else if (company %in% c('ARRAY', 'Questech Recruitment Ltd', 'EMPLOYMENT SPECIALISTS LTD', 'Precedo Healthcare', '', 'Switch Recruitment Services Ltd', 'Triumph Consultants', 'Fresh Partnership', 'W5 Recruitment', 'Recruitment Revolution', 'Excel Technical Resourcing Solutions Ltd', 'Future Select', 'Petrie Recruitment')) 
    {
      type = 'good'
    }
    else  if (company %in% c('Multilingualvacancies', 'MyUkJobs', 'nijobfinder.co.uk', 'caterer.com', 'jobs.catererandhotelkeeper.com', 'leisurejobs.com', 'Jobcentre Plus')) 
    {
      type = 'mid'
    }
    else if (company %in% c('The Works Uk Ltd', 'Castle Recruitment', 'Recruitment Direct', 'ACS Recruitment Consultants Ltd', 'Aston Taylor', 'mgi recruitment', 'Simply Recruit Ltd', 'Edustaff   Birmingham', 'Hays Specialist Recruitment', 'Towngate Personnel'))
    {
      type = 'decent'
    }
    else if (company %in% c('Forde Recruitment', 'Industrial Personnel Ltd', 'Clear Selection', 'JHR', 'Chef Results', 'Gap Personnel', 'Taskmaster', 'Red Dot Recruitment', 'Recruitment North West'))
    {
      type = 'poor'
    }
    else {
      type = 'small'
    }
    out <- append(out, type)
  }
  return(out)
}


library('DAAG')

##Suggested work from my main man Rainer
setwd('/Users/rjohnson/Documents/DS/DataScience/hw3')
# Read in separate train and test files
train <- read.csv("train.csv")
test <- read.csv("test.csv")
location_tree <- read.csv("Location_Tree2.csv")

all <- rbind( train[, c(2, 5, 6, 7, 8, 9, 12), drop=F], test[, c(2, 5, 6, 7, 8, 9, 10), drop=F])

all$CityType <- getCityTypeSal(all$LocationNormalized)
all$SourceType <- getSourceTypeSal(all$SourceName)
all$CompanyType <- getCompanyType(all$Company)
all$CombinedType <- paste(all$ContractType, all$ContractTime)
head(all)
final <- all[, c(6, 8, 9, 10,11)]
head(final)
finalX <- model.matrix(~Category+CombinedType+CityType+SourceType+CompanyType, data=final)
training <- cbind(as.data.frame(finalX[1:10000,]), train[,"SalaryNormalized", drop=F])
testing <- cbind(as.data.frame(finalX[10001:15000,]), data.frame(SalaryNormalized=NA))


model <- lm(SalaryNormalized ~ . -1, data=training)
pred <- predict(model, testing)
summary(model)
model1 <- update(model, .~.-`CategoryConsultancy Jobs`)
model2 <- update(model1, .~.-`CombinedType permanent`)
model3 <- update(model2, .~.-`CombinedTypefull_time contract`)
model4 <- update(model3, .~.-`CombinedTypefull_time permanent`)
model5 <- update(model4, .~.-`CategoryLegal Jobs`)
model6 <- update(model5, .~.-`CategoryPart time Jobs`)
model7 <- update(model6, .~.-`CombinedTypepart_time contract` )
model8 <- update(model7, .~.-`CombinedTypepart_time permanent`)
model9 <- update(model8, .~.-CompanyTypepoor)
model10 <- update(model9, .~.-`CategorySocial work Jobs`)
model11 <- update(model10, .~.-`CategoryEngineering Jobs`)
model12 <- update(model11, .~.-`CategoryTrade & Construction Jobs`)
model13 <- update(model12, .~.-`CategoryMaintenance Jobs`)
model14 <- update(model13, .~.-`CategoryPR, Advertising & Marketing Jobs`)
model15 <- update(model14, .~.-`CategoryProperty Jobs`)
model16 <- update(model15, .~.-`SourceTypesmall`)
summary(model16)
pred <- predict(model16, training)

##build Model using cross validation
modelCV <- cv.lm(df = training, form.lm= formula(SalaryNormalized ~ .-1), 
                 m=9, seed=19)
cv <- FALSE
if (cv) {
  #Show results for cv.lm.  THere is always something wierd in my last fold for this
  meanError <-mean(abs(modelCV$SalaryNormalized - modelCV$Predicted))
}
else{
  meanError <- mean(abs(training$SalaryNormalized - pred))
}

pred <- predict(model16, testing)

sink("output.txt")
cbind(testing$ID, pred)

