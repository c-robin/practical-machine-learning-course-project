---
title: "Pratical Machine Learning, Course project"
author: "Charles ROBIN"
date: "Sunday, March 22, 2015"
output: html_document
---

### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

### Assignement
We propose a machine learnign algorthim in order to predict the activity quality from activity monitors. To do so, we train a random forest over training data.

## Loading Data

```r
library(caret)
library(kernlab)
library(randomForest)
library(corrplot)

# read the csv file for training
data_training <- read.csv("./data/pml-training.csv", na.strings= c("NA",""," "))

# clean the data by removing columns with NAs etc
data_training_NAs <- apply(data_training, 2, function(x) {sum(is.na(x))})
data_training_clean <- data_training[,which(data_training_NAs == 0)]

# remove identifier columns such as name, timestamps etc
data_training_clean <- data_training_clean[8:length(data_training_clean)]
```

## Building data sets for training and cross validation. 
We use 70% for training and 30% for Cross Validation.


```r
# split the cleaned testing data into training and cross validation
inTrain <- createDataPartition(y = data_training_clean$classe, p = 0.7, list = FALSE)
training <- data_training_clean[inTrain, ]
crossval <- data_training_clean[-inTrain, ]
```

For info, here is the correlation matrix:

![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3-1.png) 

## Training a random Forest


```r
# fit a model to predict the classe using everything else as a predictor
model <- randomForest(classe ~ ., data = training)
```

## Cross-validation using the cross validation set


```r
# crossvalidate the model using the remaining 30% of data
predictCrossVal <- predict(model, crossval)
confusionMatrix(crossval$classe, predictCrossVal)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    5 1131    3    0    0
##          C    0    3 1020    3    0
##          D    0    0   12  951    1
##          E    0    0    1    6 1075
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9942         
##                  95% CI : (0.9919, 0.996)
##     No Information Rate : 0.2853         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9927         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9970   0.9974   0.9846   0.9906   0.9991
## Specificity            1.0000   0.9983   0.9988   0.9974   0.9985
## Pos Pred Value         1.0000   0.9930   0.9942   0.9865   0.9935
## Neg Pred Value         0.9988   0.9994   0.9967   0.9982   0.9998
## Prevalence             0.2853   0.1927   0.1760   0.1631   0.1828
## Detection Rate         0.2845   0.1922   0.1733   0.1616   0.1827
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9985   0.9978   0.9917   0.9940   0.9988
```

## Clean the test set and predict for the assignement submission


```r
# apply the same treatment to the final testing data
data_test <- read.csv("./data/pml-testing.csv", na.strings= c("NA",""," "))
data_test_NAs <- apply(data_test, 2, function(x) {sum(is.na(x))})
data_test_clean <- data_test[,which(data_test_NAs == 0)]
data_test_clean <- data_test_clean[8:length(data_test_clean)]

# predict the classes of the test set
predictTest <- predict(model, data_test_clean)
```
