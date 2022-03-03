## Usha V
## Choose Your Own Project 
## HarvardX: PH125.9x - Capstone Project
## https://github.com/Usha-sw/Diabetes-Prediction-Project.git

#################################################
# PIMA Indians Diabetes Prediction Project 
################################################

#### Introduction ####

## Dataset ##

# Check all necessary libraries

if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("recorrplotadr", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(caretEnsemble)) install.packages("caretEnsemble", repos = "http://cran.us.r-project.org")
if(!require(grid)) install.packages("grid", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", dependencies = TRUE, repos = "http://cran.us.r-project.org")
if(!require(ggfortify)) install.packages("ggfortify", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(funModeling)) install.packages("funModeling", repos = "http://cran.us.r-project.org")
install.packages('rminer')
library(rminer)
library(funModeling)
install.packages('corrplot')
library(corrplot)
install.packages('e1071')
library(e1071)
install.packages('Amelia')
library(Amelia)
install.packages("mice")
library(mice)

#################################################
#  Pima Indians Diabetes Detection Project
################################################

#### Data Loading ####
# Pima Indians Diabetes Dataset
# https://www./kaggle/input/pima-indians-diabetes-database/diabetes.csv'
# Loading the csv data file from my github account

data <- read.csv("https://raw.githubusercontent.com/Usha-sw/Diabetes-Prediction-Project/main/diabetes.csv")

#Setting outcome variables as categorical
data$Outcome <- factor(data$Outcome, levels = c(0,1), labels = c("False", "True"))

# General Data Info
str(data)
summary(data)

## We have 768 obs. of  9 variables. 

head(data)

describe(data)

#Convert '0' values into NA
data[, 2:7][data[, 2:7] == 0] <- NA

#visualize the missing data
missmap(data)

#Use mice package to predict missing values
library(mice)
mice_mod <- mice(data[, c("Glucose","BloodPressure","SkinThickness","Insulin","BMI")], method='rf')
mice_complete <- complete(mice_mod)

#Transfer the predicted missing values into the main data set
data$Glucose <- mice_complete$Glucose
data$BloodPressure <- mice_complete$BloodPressure
data$SkinThickness <- mice_complete$SkinThickness
data$Insulin<- mice_complete$Insulin
data$BMI <- mice_complete$BMI

#visualize the missing data
missmap(data)

## no missing values

# Check proporton of data
prop.table(table(data$Outcome))

# Distribution of the  Outcome Column
options(repr.plot.width=4, repr.plot.height=4)
ggplot(data, aes(x=Outcome))+geom_bar(fill="blue",alpha=0.5)+theme_bw()+labs(title="Distribution of Outcome")

# Plotting Numerical Data
plot_num(data %>% select (-Outcome), bins=10)

# Correlation plot
correlationMatrix <- cor(data[,1:ncol(data)-1])
heatmap(correlationMatrix)

# Find attributes that are highly corrected (ideally >0.90)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.9)
# print indexes of highly correlated attributes
print(highlyCorrelated)

# No two independent variables are highly correlated

pca_res_data <- prcomp(data[,1:ncol(data)-1], center = TRUE, scale = TRUE)
plot(pca_res_data, type="l")

# Summary of data after PCA
summary(pca_res_data)

# PC's in the transformed dataset
pca_df <- as.data.frame(pca_res_data$x)
ggplot(pca_df, aes(x=PC1, y=PC2, col=data$Outcome)) + geom_point(alpha=0.5)

# Plot of pc1 and pc2
g_pc1 <- ggplot(pca_df, aes(x=PC1, fill=data$Outcome)) + geom_density(alpha=0.25)  
g_pc2 <- ggplot(pca_df, aes(x=PC2, fill=data$Outcome)) + geom_density(alpha=0.25)  
grid.arrange(g_pc1, g_pc2, ncol=2)

# Linear Discriminant Analysis (LDA)

# Data with LDA
lda_res_data <- MASS::lda(Outcome~., data = data, center = TRUE, scale = TRUE) 
lda_res_data

#Data frame of the LDA for visualization purposes
lda_df_predict <- predict(lda_res_data, data)$x %>% as.data.frame() %>% cbind(Outcome=data$Outcome)
ggplot(lda_df_predict, aes(x=LD1, fill=diagnosis)) + geom_density(alpha=0.5)

### Model creation


# Creation of the partition 80% and 20%
set.seed(1815) 
data_sampling_index <- createDataPartition(data$Outcome, times=1, p=0.8, list = FALSE)
train_data <- data[data_sampling_index, ]
test_data <- data[-data_sampling_index, ]

prop.table(table(train_data$Outcome)) * 100
prop.table(table(test_data$Outcome)) * 100

### Naive Bayes Model

# Creation of Naive Bayes Model
model_naiveb <- naiveBayes(Outcome ~ ., data = train_data)

# Prediction
prediction_naiveb <- predict(model_naiveb, test_data)

# Confusion matrix
confusionmatrix_naiveb <- confusionMatrix(prediction_naiveb, test_data$Outcome)
confusionmatrix_naiveb

### Logistic Regression Model 

# Creation of Logistic Regression Model

model_logreg <- train(Outcome~.,
                      data=train_data,
                      method="glm")
# Prediction
prediction_logreg<- predict(model_logreg,test_data)

# Confusion matrix
confusionmatrix_logreg <- confusionMatrix(prediction_logreg, test_data$Outcome)
confusionmatrix_logreg

# Plot of top important variables
plot(varImp(model_logreg), main="Top variables - Log Regr")

### Random Forest Model

# Creation of Random Forest Model
model_randomforest <- train(Outcome~.,
                            data=train_data,
                            method="rf")
# Prediction
prediction_randomforest <- predict(model_randomforest, test_data)

# Confusion matrix
confusionmatrix_randomforest <- confusionMatrix(prediction_randomforest, test_data$Outcome)
confusionmatrix_randomforest

# Plot of top important variables
plot(varImp(model_randomforest), main="Top variables- Random Forest")


### K Nearest Neighbor (KNN) Model

# Creation of K Nearest Neighbor (KNN) Model
model_knn <- train(Outcome~.,
                   data=train_data,
                   method="knn",
                   tuneLength=5) #The tuneLength parameter tells the algorithm to try different default values for the main parameter
                   #In this case we used 5 default values
                   
# Prediction
prediction_knn <- predict(model_knn, test_data)

# Confusion matrix        
confusionmatrix_knn <- confusionMatrix(prediction_knn, test_data$Outcome)
confusionmatrix_knn

# Plot of top important variables
plot(varImp(model_knn), main="Top variables - KNN")

### Neural Network with PCA Model

# Creation of Neural Network Model
model_nnet_pca <- train(Outcome~.,
                        data=train_data,
                        method="nnet",
                        trace=FALSE)

# Prediction
prediction_nnet_pca <- predict(model_nnet_pca, test_data)

# Confusion matrix
confusionmatrix_nnet_pca <- confusionMatrix(prediction_nnet_pca, test_data$Outcome)
confusionmatrix_nnet_pca

# Plot of top important variables
plot(varImp(model_nnet_pca), main="Top variables - NNET PCA")

### Neural Network with LDA Model

# Creation of training set and test set with LDA modified data
train_data_lda <- lda_df_predict[data_sampling_index, ]
test_data_lda <- lda_df_predict[-data_sampling_index, ]

# Creation of Neural Network with LDA Model
model_nnet_lda <- train(Outcome~.,
                        data = train_data_lda,
                        method="nnet",
                        trace=FALSE)
# Prediction
prediction_nnet_lda <- predict(model_nnet_lda, test_data_lda)
# Confusion matrix
confusionmatrix_nnet_lda <- confusionMatrix(prediction_nnet_lda, test_data_lda$Outcome)
confusionmatrix_nnet_lda

# Results

# Confusion matrix of the models
confusionmatrix_list <- list(
  Naive_Bayes=confusionmatrix_naiveb, 
  Logistic_regr=confusionmatrix_logreg,
  Random_Forest=confusionmatrix_randomforest,
  KNN=confusionmatrix_knn,
  Neural_PCA=confusionmatrix_nnet_pca,
  Neural_LDA=confusionmatrix_nnet_lda)   
confusionmatrix_list_results <- sapply(confusionmatrix_list, function(x) x$byClass)
confusionmatrix_list_results %>% knitr::kable()

# Discussion

# Find the best result for each metric
confusionmatrix_results_max <- apply(confusionmatrix_list_results, 1, which.is.max)
output_report <- data.frame(metric=names(confusionmatrix_results_max), 
                            best_model=colnames(confusionmatrix_list_results)
                            [confusionmatrix_results_max],
                            value=mapply(function(x,y) 
                            {confusionmatrix_list_results[x,y]}, 
                            names(confusionmatrix_results_max), 
                            confusionmatrix_results_max))
rownames(output_report) <- NULL
output_report

# Appendix - Enviroment
# Print system information
print("Operating System:")
R.version
