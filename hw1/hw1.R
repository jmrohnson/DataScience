collegeData <- read.csv("collegesClean.csv", sep=',', h=F)
colnames(collegeData) <- c('FICE', 'Name', 'State', 'Type', 'avgSalFull', 'avgSalAssc', 'avgSalAsst', 'avgSalAll', 'avgCompFull', 'avgCompAssc', 'avgCompAsst', 'avgCompAll', 'numFull', 'numAssc', 'numAsst', 'numInstr', 'numFacAll')
collegeData$FullPerc <- apply(collegeData, 1, function(row) as.numeric(row[13]) / as.numeric(row[17]))
collegeData$AsscPerc <- apply(collegeData, 1, function(row) as.numeric(row[14]) / as.numeric(row[17]))
collegeData$AsstPerc <- apply(collegeData, 1, function(row) as.numeric(row[15]) / as.numeric(row[17]))
collegeData$InstrPerc <- apply(collegeData, 1, function(row) as.numeric(row[16]) / as.numeric(row[17]))
fit <- lm(avgSalFull~ avgSalAssc + avgSalAsst + numFull + numAssc + numAsst + numInstr, data=collegeData)
fit1 <- update(fit1, .~.-numAssc)
fit2 <- update(fit2, .~.-numAsst)
fit3 <- update(fit3, .~.-numInstr)

#Turns out fit1 seemed to be best
summary(fit1)

plot(resid(fit1))
qqnorm(resid(fit1))

# the ggplot data visualization
ggplot(data=collegeData, aes(x=FullPerc, y=avgSalFull, color=Type)) + geom_point() + geom_smooth()


#Bonus -- this doesn't work, I believe I've done something wrong with the constants as I get the same response as avgSalFull~numFull
#Do I need to assign each to a new variable?
overFit <- lm(avgSalFull ~ numFull + numFull^2 + numFull^3 + numFull^4 + numFull^5, data=collegeData)
summary(overFit)