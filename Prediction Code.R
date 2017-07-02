
#Setting up the Environment

setwd("C:/Work/Others/Competitions/AV Loan Prediction")
library(caret)
library(dplyr)


#Reading the Training Data. Loan ID is removed from the dataset as it is never used in the algorithm
trainingData <- read.csv("Train Set.csv",na.strings=c("NA",""))
dataForPrediction <- select(trainingData,-Loan_ID)

#Credit History is used as a factorvariable and it forms as a routing factor in the algorithm. Hence its coded as "Good" or "Bad"
dataForPrediction <- mutate(dataForPrediction, Credit_History = ifelse(Credit_History == '1',"Good","Bad"))
dataForPrediction <- mutate(dataForPrediction, Credit_History = as.factor(Credit_History))

#Imputing model for numberical variabes using the knn impute function
imputeModel <- preProcess(dataForPrediction,method = "knnImpute") ##aLSO Centers and scales data
imputedPredictionData <- predict(imputeModel,dataForPrediction)

#Categorical variables which are not provided are coded as "Not Provided"
imputedPredictionData <- mutate(imputedPredictionData, Gender = as.character(Gender),
                                Married = as.character(Married),
                                Dependents = as.character(Dependents),
                                Education = as.character(Education),
                                Self_Employed = as.character(Self_Employed),
                                Property_Area = as.character(Property_Area),
                                Loan_Status = as.character(Loan_Status),
                                Credit_History = as.character(Credit_History))
imputedPredictionData[is.na(imputedPredictionData)] <- "Not Provided"
imputedPredictionData <- mutate(imputedPredictionData, Gender = as.factor(Gender),
                                Married = as.factor(Married),
                                Dependents = as.factor(Dependents),
                                Education = as.factor(Education),
                                Self_Employed = as.factor(Self_Employed),
                                Property_Area = as.factor(Property_Area),
                                Loan_Status = as.factor(Loan_Status),
                                Credit_History = as.factor(Credit_History))


#Setting train control parameters - cross validation of 10 selections repeated 3 times
train_control <- trainControl(method="repeatedcv", number=10, repeats=3)





#Training model using decision tree when Credit History is Good
goodHistoryData <- filter(imputedPredictionData, Credit_History == "Good")
treeCreditGood <- train(Loan_Status~.,data=goodHistoryData,trControl = train_control, method = "rpart", control=rpart.control(minsplit=15, cp = 0.001))


#Training model using Decision Tree when Credit History is not Good
notGoodHistoryData <- filter(imputedPredictionData, Credit_History != "Good")
treeCreditNotGood <- train(Loan_Status~.,data=notGoodHistoryData, method = "rpart", control=rpart.control(minsplit=5))





###Predicting the Testing data
#Getting Test Data
testingData <- read.csv("Testing Set.csv",na.strings=c("NA",""))
preProcessData <- mutate(testingData, Credit_History = ifelse(Credit_History == '1',"Good","Bad"))
preProcessData <- mutate(preProcessData, Credit_History = as.factor(Credit_History))


#Apply pre processing to mimic data used for training
imputeModelPred <- preProcess(preProcessData,method = "knnImpute") ##aLSO Centers and scales data
preProcessData <- predict(imputeModel,preProcessData)
preProcessData <- mutate(preProcessData, Gender = as.character(Gender),
                                Married = as.character(Married),
                                Dependents = as.character(Dependents),
                                Education = as.character(Education),
                                Self_Employed = as.character(Self_Employed),
                                Property_Area = as.character(Property_Area),
                                
                                Credit_History = as.character(Credit_History))
preProcessData[is.na(preProcessData)] <- "Not Provided"
preProcessData <- mutate(preProcessData, Gender = as.factor(Gender),
                                Married = as.factor(Married),
                                Dependents = as.factor(Dependents),
                                Education = as.factor(Education),
                                Self_Employed = as.factor(Self_Employed),
                                Property_Area = as.factor(Property_Area),
                                
                                Credit_History = as.factor(Credit_History))

#Prediction when Credit History iS Good
goodCreditData <- filter(preProcessData,Credit_History == "Good")
predictions <- predict(treeCreditGood,goodCreditData)
goodCreditData <- mutate(goodCreditData, Loan_Status_Custom = predictions)


#Prediction when credit history is Bad
badCreditData <- filter(preProcessData,Credit_History != "Good")
predictions <- predict(treeCreditNotGood,badCreditData)
badCreditData <- mutate(badCreditData, Loan_Status_Custom = predictions)


#Joining the predicted values and exporting predictions
predictionData <- rbind(goodCreditData,badCreditData)
predictionOutputcustom <- select(predictionData,Loan_ID,Loan_Status_Custom)
#write.csv(predictionOutput, "predictions.csv")



#Random Forest
modelRF <- train(Loan_Status~.,data=imputedPredictionData,trControl = train_control, method = "rf",control=rpart.control(minsplit=5, cp=0.001))
predictions <- predict(modelRF,preProcessData)
predictionOutputRF <- mutate(preProcessData ,Loan_Status_RF = predictions)
predictionOutputRF <- select(predictionOutputRF,Loan_ID, Loan_Status_RF)

#Decision Tree
modelRP <- train(Loan_Status~.,data=imputedPredictionData,trControl = train_control, method = "rpart",control=rpart.control(minsplit=10, cp=0.001))
predictions <- predict(modelRP,preProcessData)
predictionOutputRP <- mutate(preProcessData ,Loan_Status_DT = predictions)
predictionOutputRP <- predictionOutputRP %>%select(Loan_ID,Loan_Status_DT)


#Joining Predicted Data
customRFJoin <- inner_join(predictionOutputcustom,predictionOutputRF)
fullJoin <- inner_join(customRFJoin,predictionOutputRP)
fullJoin <- mutate(fullJoin,custom = ifelse(Loan_Status_Custom == 'Y',1,0), RF = ifelse(Loan_Status_RF == 'Y',1,0),
                   DT = ifelse(Loan_Status_DT == 'Y',1,0))
fullJoin <- mutate(fullJoin, Loan_Status = ifelse(custom+RF+DT >=1,'Y','N'))
write.csv(select(fullJoin,Loan_ID, Loan_Status),"Predictions.csv")
