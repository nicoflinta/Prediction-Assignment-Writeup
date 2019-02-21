# Prediction-Assignment-Writeup
Prediction Assignment Writeup

author: "Nicolas Flinta"
date: "21 Feb 2019"

# Executive Summary

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. 

### Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Exploratory Data Analysis

Download the data.


```r
## Download the testing and training datasets and open them as data tables
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.5.1
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.5.1
```

```r
library(ggplot2)

setwd("C:/Users/601459542/OneDrive - BT Plc/Documents/Visual Studio 2017/BIDs/Data Science Coursera/Prediction Assignment Writeup")

## Download Files 
# fileUrl_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
# fileUrl_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
# download.file(fileUrl_train, destfile = paste0(getwd(), '/pml-training.csv'))
# download.file(fileUrl_test, destfile = paste0(getwd(), '/pml-testing.csv'))

## Load traininf data into data frame
activity_data_train_all <- read.csv("pml-training.csv")
```

Split training data set (75/25) for validation purposes.


```r
## Save some data observations as validation set
dpart <- createDataPartition(activity_data_train_all$classe, p = 0.25, list = F)
activity_data_train <- activity_data_train_all[-dpart,]
activity_data_validation <- activity_data_train_all[dpart,]


## Summarise the data for training model.
# summary(activity_data_train) <<< Commented to reduce the lenght of the markdown.
```

Remove columns with NA and factors (by keeping only numerical variables) as they will not be related to the Classe variable.


```r
## Delete all NA columns from the data
col_with_na <- numeric()
for (i in 1:length(activity_data_train)) {
    if (any(is.na(activity_data_train[i]))) { col_with_na <- append(col_with_na, i) }

}
activity_data_train <- activity_data_train[-col_with_na]

## Keep only numerical variables and append the classe variable to the data frame
is.num <- sapply(activity_data_train, is.numeric)
num.df <- activity_data_train[, is.num]

## Remove participant names and dates
num.df <- num.df[-(1:4)]
num.df$classe <- activity_data_train$classe
activity_data_train <- num.df
```

## Model Training

### Model 0
For classification and regression using Classification Tree (method rpart).

```r
set.seed(12345)
rf_model_0 <- train(classe ~ ., data = activity_data_train, method = "rpart")
print(rf_model_0)
```

```
## CART 
## 
## 14715 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 14715, 14715, 14715, 14715, 14715, 14715, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa     
##   0.03466287  0.5000302  0.34728962
##   0.06014562  0.3928346  0.16915313
##   0.11623932  0.3338122  0.07516695
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.03466287.
```

Accuracy of the model is *51.56%*, which is a good place to start but not great for predicting the 'classe' dependent variable.

### Model 1
Second model I used RandomForest (method = "rf") in the Caret package. Also using cross validation with 5 k-folds.


```r
set.seed(12345)
rf_model_1 <- train(classe ~ ., data = activity_data_train, method = "rf",
                trControl = trainControl(method = "cv", number = 5),
                prox = TRUE, allowParallel = TRUE)
print(rf_model_1)
```

```
## Random Forest 
## 
## 14715 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 11772, 11772, 11772, 11772, 11772 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9910975  0.9887372
##   27    0.9919810  0.9898553
##   52    0.9857968  0.9820317
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

The second model using Random Forest with cross validations gives me a 99.15% accuracy (I wish my real life models were like this!).

## Model Testing

### Model 0

```r
prediction_0 <- predict(rf_model_0, activity_data_validation)
# You can use the prediction to compute the confusion matrix and see the accuracy score
confusionMatrix(prediction_0, activity_data_validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1274  399  389  368  132
##          B   26  303   23  139  117
##          C   91  248  444  297  256
##          D    0    0    0    0    0
##          E    4    0    0    0  397
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4928          
##                  95% CI : (0.4787, 0.5069)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.337           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9133  0.31895  0.51869   0.0000  0.44013
## Specificity            0.6333  0.92292  0.77981   1.0000  0.99900
## Pos Pred Value         0.4973  0.49836  0.33234      NaN  0.99002
## Neg Pred Value         0.9484  0.84950  0.88463   0.8362  0.88793
## Prevalence             0.2843  0.19360  0.17444   0.1638  0.18382
## Detection Rate         0.2596  0.06175  0.09048   0.0000  0.08090
## Detection Prevalence   0.5221  0.12390  0.27226   0.0000  0.08172
## Balanced Accuracy      0.7733  0.62093  0.64925   0.5000  0.71957
```

Using the Validation data set we see the accuracy is 49% and that the model performs better for certain classes but not great overall.

### Model 1

```r
prediction_1 <- predict(rf_model_1, activity_data_validation)
# You can use the prediction to compute the confusion matrix and see the accuracy score
confusionMatrix(prediction_1, activity_data_validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    7    0    0    0
##          B    0  942    1    0    1
##          C    0    1  848   12    0
##          D    0    0    7  791    5
##          E    0    0    0    1  896
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9929         
##                  95% CI : (0.9901, 0.995)
##     No Information Rate : 0.2843         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.991          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9916   0.9907   0.9838   0.9933
## Specificity            0.9980   0.9995   0.9968   0.9971   0.9998
## Pos Pred Value         0.9950   0.9979   0.9849   0.9851   0.9989
## Neg Pred Value         1.0000   0.9980   0.9980   0.9968   0.9985
## Prevalence             0.2843   0.1936   0.1744   0.1638   0.1838
## Detection Rate         0.2843   0.1920   0.1728   0.1612   0.1826
## Detection Prevalence   0.2857   0.1924   0.1755   0.1636   0.1828
## Balanced Accuracy      0.9990   0.9955   0.9937   0.9905   0.9965
```

WOW! 99.29% accuracy and a very well balanced accuracy accross all classes.

## Prediction on Test set

Predicting the class for each 'problem_id' in the test dataset.


```r
activity_data_test <- read.csv("pml-testing.csv")
prediction_answers_1 <- predict(rf_model_1, activity_data_test)
validation_results <- data.frame(problem_id = activity_data_test$problem_id, predicted = prediction_answers_1)
print(validation_results)
```

```
##    problem_id predicted
## 1           1         B
## 2           2         A
## 3           3         B
## 4           4         A
## 5           5         A
## 6           6         E
## 7           7         D
## 8           8         B
## 9           9         A
## 10         10         A
## 11         11         B
## 12         12         C
## 13         13         B
## 14         14         A
## 15         15         E
## 16         16         E
## 17         17         A
## 18         18         B
## 19         19         B
## 20         20         B
```
