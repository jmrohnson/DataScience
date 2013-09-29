#Working with classifying the IRIS data
library(class)
library(ggplot2)

Myknn <- function(data, labels, test.index, max.k) {
  train.data <- data[-test.index,]
  test.data <- data[test.index,]
  train.labels<- as.factor(as.matrix(labels)[-test.index,])
  test.labels<- as.factor(as.matrix(labels)[test.index,]) # this will apply the labels to the test/traiing sets
  err.rates <- data.frame() # initialzie results object
  for (k in 1:max.k) # start a for loop to see best k
  {
    knn.fit <- knn(train=train.data,  # train set
                   test=test.data,    # test set
                   cl = train.labels, # true labels
                   k = k              # number of Nearest Neighbors
    )
    this.err <- sum(test.labels != knn.fit) / length(test.labels) #store gzn error
    err.rates <- rbind(err.rates, this.err) # append to error results
  }
  
  return(err.rates)
}

knn.nfold <- function(data, labels, n, max.k) 
{
  set.seed(3)
  N <- nrow(data)
  test.pct = 1/n
  step = test.pct*N
  randIndices <- sample(1:N, N)
  #err.rates <-data.frame() # initialzie results object
  for (i in 1:n-1)
  {
    start <- i*step
    end <- (i+1)*step
    test.indices <- randIndices[start:end]
    c <- Myknn(data, labels, test.indices, max.k)
    if (i == 0){
      err.rates <- c
    }
    else {
      err.rates <- cbind(err.rates, c)
    }
  }
  return(err.rates)
}


iris <- NULL
err.rates <-NULL
labels <-NULL
results <- NULL
test.indices <- NULL
test.data <- NULL
train.data <- NULL
data(iris)
labels <- iris$Species
iris$Species<-NULL
err.rates <- data.frame()
results <- data.frame()
max.k <- 100
err.rates <- knn.nfold(iris, labels,5, max.k)
err.rates$ave <- apply(err.rates, 1, function(row) mean(row))

results <- data.frame(1:max.k, err.rates$ave) # create resulsts summary data frame
names(results) <- c('K', 'err.rate')
#create a sweet plot
title <- paste('knn results', sep='') # give us a title for our plot
results.plot <- ggplot(data = results, aes(x=K, y=err.rate)) + geom_point() + geom_line()
results.plot <- results.plot + ggtitle(title)
results.plot