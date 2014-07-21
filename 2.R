library(caret)

training     <- read.csv("pml-training.csv") # training data
test         <- read.csv("pml-testing.csv" ) # test     data

# remove index column (not a predictor)
training$X   <- NULL
test$X       <- NULL

# remove raw timestamp columns
#data$raw_timestamp_part_1 <- NULL
#validation$raw_timestamp_part_1  <- NULL
#data$raw_timestamp_part_2 <- NULL
#validation$raw_timestamp_part_2  <- NULL

# remove predictors where more than 97% of the values are NA
percentNA  <- as.vector(colMeans(is.na(training)))
colsNA     <- which(percentNA > 0.97)
training   <- training[, -colsNA]
test       <- test    [, -colsNA]

# remove near-zero variance predictors 
nsv        <- nearZeroVar(training)
training   <- training[, -nsv]
test       <- test    [, -nsv]

# find which predictors are factor predictors
colsFactor <- as.vector(which(sapply(training, is.factor)))

# ensure reproducability
set.seed(123)

# further reduce the number of predictors using Principal Component Analysis on training data, retaining 99% of the variance
preProc    <- preProcess(training[,-colsFactor], method= "pca", thresh= 0.99)

# replace training and test sets with PCA results; bind removed factor columns
training   <- cbind(predict(preProc, training[, -colsFactor]), training[colsFactor])
test       <- cbind(predict(preProc, test    [, -colsFactor]), test    [colsFactor])

# 10-fold cross-validation random-forest training
fitControl <- trainControl(method= "cv", number= 10, repeats= 1, verboseIter= T)
modFit     <- train(classe ~ ., method= "rf" , data= training, trControl= fitControl)

p          <- predict(modFit, training)
confusionMatrix(p, training$classe)

# postResample(p, training$classe)

q          <- predict(modFit, test)

# get structure of trees k in the forest
# getTree(modFit$finalModel, k= 2)
