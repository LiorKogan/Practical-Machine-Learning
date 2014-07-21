library(caret)

data       <- read.csv("pml-training.csv") # training + test data
validation <- read.csv("pml-testing.csv" ) # validatation    data

# remove index column (not a predictor)
data$X       <- NULL
validation$X <- NULL

# remove raw timestamp columns
#data$raw_timestamp_part_1 <- NULL
#validation$raw_timestamp_part_1  <- NULL
#data$raw_timestamp_part_2 <- NULL
#validation$raw_timestamp_part_2  <- NULL

# remove predictors where more than 97% of the values are NA
percentNA  <- as.vector(colMeans(is.na(data)))
colsNA     <- which(percentNA > 0.97)
data       <- data      [, -colsNA]
validation <- validation[, -colsNA]

# remove near-zero variance predictors 
nsv        <- nearZeroVar(data)
data       <- data      [, -nsv]
validation <- validation[, -nsv]

# find which predictors are factor predictors
colsFactor <- as.vector(which(sapply(data, is.factor)))

#cors       <- abs(cor(data[,-colsFactor]))
#diag(cors) <- 0

# sapply(1:58, function(x) skewness(data[, x]))

set.seed(123) # ensure reproducability

inTrain    <- createDataPartition(data$classe, p= 0.80, list= F)
training   <- data[ inTrain,]
test       <- data[-inTrain,]

# further reduce the number of predictors using Principal Component Analysis on training data, retaining 99% of the variance
preProc    <- preProcess(training[,-colsFactor], method= "pca", thresh= 0.99)

# replace training, test and validation sets with PCA results; bind removed factor columns
training   <- cbind(predict(preProc, training  [,-colsFactor]), training  [colsFactor])
test       <- cbind(predict(preProc, test      [,-colsFactor]), test      [colsFactor])
validation <- cbind(predict(preProc, validation[,-colsFactor]), validation[colsFactor])

# 10-fold cross-validation training
fitControl <- trainControl(method= "cv", number= 10, repeats= 1, verboseIter= T)


modFit1    <- train(classe ~ ., method= "rf" , data= training, trControl= fitControl)
modFit2    <- train(classe ~ ., method= "gbm", data= training, trControl= fitControl)

p1 <- predict(modFit1, test)
p2 <- predict(modFit2, test)

confusionMatrix(p1, test$classe)$overall[1]
confusionMatrix(p2, test$classe)$overall[1]

q1 <- predict(modFit1, validation)
q2 <- predict(modFit2, validation)

confusionMatrix(q1, validation$classe)$overall[1]
confusionMatrix(q2, validation$classe)$overall[1]

# get structure of the randomForest tree k
# getTree(modFit1$finalModel, k= 2)
