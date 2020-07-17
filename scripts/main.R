### Classification Modelling ###
### Objective: Target customers highly likely to purchase a new vehicle ###

# Clear environment #
rm(list = ls())

# Packages
library(dplyr)
library(corrplot)
library(rpart)
library(rpart.plot)
library(MASS)
library(ggcorrplot)
library(ggplot2)
library(GGally)
library(pROC)
library(ROCR)
library(caret)
library(car)
library(pdp)
library(DMwR)
library(rattle)
library(funModeling)

# Set working directory
setwd("~/GitHub/classification/data")

# Load Dataset
repurchase_training <- read.csv("repurchase_training.csv")

#### Exploration Data Analysis ####
str(repurchase_training)
head(repurchase_training)
tail(repurchase_training)
summary(repurchase_training) # Mean of Target = 0.02681
df_status(repurchase_training) # No NAs, 97.32% of 0s in Target column

# Function to order categorical variable in descending order
reorder_size <- function(x) {
  factor(x, levels = names(sort(table(x), decreasing = TRUE)))
}

# Rename target column for data hygiene
repurchase_training <- repurchase_training %>% rename(target = Target)

# Target as factor
repurchase_training$target <- as.factor(repurchase_training$target)

# Drop ID column 
repurchase_training <- repurchase_training[, -1]

# Explore categorical variables
table(reorder_size(repurchase_training$gender))
table(reorder_size(repurchase_training$age_band))
table(reorder_size(repurchase_training$car_model))
table(reorder_size(repurchase_training$car_segment))
table(repurchase_training$gender, repurchase_training$age_band)

# Visually exploring dataset
# Categorical variables
ggplot(data = repurchase_training) + geom_bar(mapping = aes(x = reorder_size(gender), fill = factor(target, levels = c("1", "0"))))

ggplot(data = repurchase_training) + geom_bar(mapping = aes(x = reorder_size(age_band), fill = factor(target, levels = c("1", "0"))))

ggplot(data = repurchase_training) + geom_bar(mapping = aes(x = reorder_size(car_model), fill = factor(target, levels = c("1", "0"))))

ggplot(data = repurchase_training) + geom_bar(mapping = aes(x = reorder_size(car_segment), fill = factor(target, levels = c("1", "0"))))

# Interesting numerical values showing target = 1 variance
ggplot(data = repurchase_training) + geom_bar(mapping = aes(x = reorder_size(as.factor(age_of_vehicle_years)), fill = factor(target, levels = c("1", "0"))))

ggplot(data = repurchase_training) + geom_bar(mapping = aes(x = reorder_size(as.factor(sched_serv_warr)), fill = factor(target, levels = c("1", "0"))))

# For numerical variables, the data has been split in deciles so it is evenly split into categories from 1 to 10. No outliers.

# Drop age_band, gender due to high volume of missing values, 85.5% & 52.77%
repurchase_training <- repurchase_training[, -c(2,3)]

# Check for multicollinearity - drops non numeric variables automatically
ggcorr(repurchase_training, method = "pairwise", label = TRUE)

# sched_serv_paid, non_sched_serv_paid, sched_serv_warr, non_sched_serv_warr, total_paid_services show multicollinearity and will not be used to build the logistic regression model

# Set seed for reproducibility
set.seed(42) 

#### Train test split ####

train_binary = createDataPartition(y = repurchase_training$target, p = 0.7, list = F)
training_binary = repurchase_training[train_binary, ]
testing_binary = repurchase_training[-train_binary, ]

# check if train/test is representative of the data set
tbl_train <- table(training_binary$target)
tbl_train

tbl_test <- table(testing_binary$target)
tbl_test

# in percentages
tbl_train_prop <- prop.table(tbl_train)
tbl_train_prop

tbl_test_prop <- prop.table(tbl_test)
tbl_test_prop

# Compare to overall - train/test split is not balanced. 97.3% "No" and 2.7% "Yes"
repurchase_prop <- prop.table(table(repurchase_training$target))
repurchase_prop

#### Logistic Regression model ####
# train model with all variables except the ones showing multicollinearity
reg_model <- glm(target ~ car_model + car_segment + age_of_vehicle_years + total_services + 
                   mth_since_last_serv + annualised_mileage + num_dealers_visited + 
                   num_serv_dealer_purchased, data = training_binary, family = binomial)

# analyse results
summary(reg_model)

# perform stepwise selection to keep most impactful variables
step <- stepAIC(reg_model, direction="both", trace=FALSE)
step$anova

# train final model by removing car_segment which was not significant based on stepwise selection
reg_model <- glm(target ~ car_model + age_of_vehicle_years + total_services + 
                         mth_since_last_serv + annualised_mileage + num_dealers_visited + 
                         num_serv_dealer_purchased, data = training_binary, family = binomial)

# Prediction object on the training data
training_binary$probability <- predict(reg_model, newdata = training_binary[, -1], type = "response")
training_binary$predictions <- ifelse(training_binary$probability > 0.02681, 1, 0)
train_pred = prediction(training_binary$probability, training_binary$target)

# Prediction object on the testing data
testing_binary$probability <- predict(reg_model, newdata = testing_binary[,-1], type = "response")
testing_binary$predictions <- ifelse(testing_binary$probability > 0.02681, 1, 0)

test_pred = prediction(testing_binary$probability, testing_binary$target)

testing_binary$predictions <- as.factor(testing_binary$predictions)
testing_binary$target <- as.factor(testing_binary$target)

# Confusion matrix
confusionMatrix(data = testing_binary$predictions, reference = testing_binary$target,
                mode = "everything", positive="1")

# tpr and fpr for training
train_tpr_fpr <- performance(train_pred, "tpr","fpr")
train_auc <- performance(train_pred, "auc")

# tpr and fpr for testing
test_tpr_fpr <- performance(test_pred, "tpr","fpr")
test_auc <- performance(test_pred, "auc")

# Plot the tpr and fpr gains chart ROC for both testing and training data
plot(test_tpr_fpr, main="Testing and Training ROC Curves", col = "blue")
plot(train_tpr_fpr, add = T, col = "red")
legend("bottomright", legend = c("Training","Testing"), col = c("red","blue"), lty = 1, lwd = 2)
abline(0,1, col = "darkgray")
grid()


# AUC figures
train_auc <- unlist(slot(train_auc, "y.values"))
train_auc

# Area under the ROC curve
test_auc <- unlist(slot(test_auc, "y.values"))
test_auc

sens <- performance(test_pred, "sens")
spec <- performance(test_pred, "spec")

# Sensitivity Specificity Chart
plot(sens, 
     main = "Sensitivity Specificity Chart", type = "l", col = "red", lwd = 2, 
     xlim = c(0,1), ylim = c(0,1), 
     ylab = "Values")
axis(side = 1, at = seq(0, 1, 0.1))
axis(side = 2, at = seq(0, 1, 0.1))
plot(spec, add = T, col = "blue", lwd = 2, 
     xlim = c(0,1), ylim = c(0,1)
)

legend("bottomright", legend = c("Sensitivity","Specificity"), col = c("red", "blue"), lty = 1, lwd = 2)
abline(h = seq(0, 1, 0.1), v = seq(0, 1, 0.1), col="gray", lty=3)

# Determining the optimal cutoff
test_sens_spec <- performance(test_pred, "sens","spec")

# Create dataframe
threshold_df <- data.frame(cut = test_sens_spec@alpha.values[[1]], 
                           sens = test_sens_spec@x.values[[1]],
                           spec = test_sens_spec@y.values[[1]])

# Determine max of sensitivity + specifcity
which.max(threshold_df$sens + threshold_df$spec)

# Assign threshold to variable
threshold <- threshold_df[which.max(threshold_df$sens + threshold_df$spec), "cut"]

# Threshold
threshold

testing_binary$predictions <- ifelse(testing_binary$probability > threshold, 1, 0)
testing_binary$predictions <- as.factor(testing_binary$predictions)

# New confusion matrix after using optimum cutoff
confusionMatrix(data = testing_binary$predictions, reference = testing_binary$target,
                mode = "everything", positive="1")

# New AUC after using optimum cutoff
# tpr and fpr for testing
test_tpr_fpr <- performance(test_pred, "tpr","fpr")
test_auc <- performance(test_pred, "auc")

test_auc <- unlist(slot(test_auc, "y.values"))
test_auc

#### Decision Tree model ####

# Reset Training and Testing set
training_binary <- training_binary[, -c(16:17)]
testing_binary <- testing_binary[, -c(16:17)]

# Create decision tree model with cross-validation + smote to balance dataset
ctrl <- trainControl(method = "cv",
                     number = 5,
                     search ="random",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     allowParallel = TRUE,
                     sampling = "smote")

# Change levels of dependent variable to run through train function
levels(training_binary$target) <- c("No", "Yes")

# Build Decision Tree Model
rpart_model <- train(target ~ ., data = training_binary, 
                     method = "rpart",
                     metric = "ROC",
                     trControl = ctrl)

# View model
rpart_model$results # Look for CP to prune tree

# Prune Tree to avoid overfitting using cp 0.0027
rpart_model <- train(target ~ ., data = training_binary, 
                     method = "rpart",
                     metric = "ROC",
                     trControl = ctrl,
                     control=rpart.control(minsplit=2, cp=0.0027))

# Plot Decision Tree
fancyRpartPlot(rpart_model$finalModel)

# Predict on test set
testing_binary$probability <- predict(rpart_model,testing_binary[,-1],type="prob")
test_pred <- prediction(testing_binary$probability[,2], testing_binary$target)

# Check confusion matrix - 2.4% out of 2.7% of "Yes/1" are captured
confusionMatrix(rpart_model,
                mode = "everything", positive="Yes")


# Sensitivity-Specificity Graphs
sens <- performance(test_pred, "sens")
spec <- performance(test_pred, "spec")

plot(sens, 
     main = "Sensitivity Specificity Chart", type = "l", col = "red", lwd = 2, 
     xlim = c(0,1), ylim = c(0,1), 
     ylab = "Values")
axis(side = 1, at = seq(0, 1, 0.1))
axis(side = 2, at = seq(0, 1, 0.1))
plot(spec, add = T, col = "blue", lwd = 2, 
     xlim = c(0,1), ylim = c(0,1)
)

legend("bottomright", legend = c("Sensitivity","Specificity"), col = c("red", "blue"), lty = 1, lwd = 2)
abline(h = seq(0, 1, 0.1), v = seq(0, 1, 0.1), col="gray", lty=3)


# Determining the optimal cutoff
test_sens_spec <- performance(test_pred, "sens","spec")

# Create dataframe
threshold_df <- data.frame(cut = test_sens_spec@alpha.values[[1]], 
                           sens = test_sens_spec@x.values[[1]],
                           spec = test_sens_spec@y.values[[1]])

# Determine max of sensitivity + specifcity and save as threshold value
which.max(threshold_df$sens + threshold_df$spec)
threshold <- threshold_df[which.max(threshold_df$sens + threshold_df$spec), "cut"]

# Threshold
threshold

## Final prediction with optimal cutoff
testing_binary$prediction_cutoff <- "0"
testing_binary[testing_binary$probability[,2] >= threshold, "prediction_cutoff"] <- "1"
testing_binary$prediction_cutoff <- as.factor(testing_binary$prediction_cutoff)
levels(testing_binary$prediction_cutoff) <- c("0","1")
testing_binary$probability = testing_binary$probability[,2]

# Confusion matrix Decision Tree Model
confusionMatrix(data = testing_binary$prediction_cutoff, reference = testing_binary$target,
                mode = "everything", positive="1")


# Area under the ROC curve Decision Tree Model
test_auc <- performance(test_pred, "auc")
test_auc <- unlist(slot(test_auc, "y.values"))
test_auc

#### Variable importance and partial dependency plots ####
Imp <- varImp(rpart_model)
tmp <- data.frame(
  overall = Imp$importance$Overall,
  name = row.names(Imp$importance)
)
topFiveImp <- head(tmp[order(tmp$overall, decreasing = T),], 5)
rm("tmp")

# Partial dependency plot
for (name in topFiveImp$name) {
  print(name)
  plot(partial(rpart_model, pred.var = name, plot = TRUE, rug = TRUE,
               type="classification", prob=TRUE, which.class = "Yes", train=training_binary))
}


#### Predict on validation set using Decision tree model ####

# load file for prediction
repurchase_validation <- read.csv("repurchase_validation.csv")


# predict
repurchase_validation$probability <- predict(rpart_model,repurchase_validation,type="prob")
repurchase_validation$prediction <-"0"
repurchase_validation[repurchase_validation$probability[,2] >= threshold, "prediction"] <- "1"
repurchase_validation$prediction <- as.factor(repurchase_validation$prediction)
levels(repurchase_validation$prediction) <- c("0","1")
repurchase_validation$probability <- repurchase_validation$probability[,2]

# create dataframe for file export

validation <- data.frame(ID = repurchase_validation$ID,
                         target_probability = repurchase_validation$probability,
                         target_class = repurchase_validation$prediction)

# export validation as csv
# write.csv(validation, "repurchase_validation_predictions.csv", row.names=FALSE)


