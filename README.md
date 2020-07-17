# emailingclassification
 
# MAILING CAMPAIGN

The below report is formatted in a way that can be shared with internal stakeholders of the business and the code (using R) for the data exploration, logistic regression and decision tree are available in the [scripts folder](https://github.com/SandraLeriche/classification/tree/master/scripts).

Data: [repurchase_training](https://github.com/SandraLeriche/classification/blob/master/data/repurchase_training.csv), [repurchase_validation](https://github.com/SandraLeriche/classification/blob/master/data/repurchase_validation.csv)

### BUSINESS OBJECTIVES

The objective of this analysis is to optimise the next communication campaign to target customers who are most likely to purchase a new vehicle.

**Assumptions, limitations and risks**
It is assumed that data recorded about customers who have purchased in the past will provide some insights on future purchases.

There are a few risks to consider when sending a mailing campaign:

- Customers who are not the target and do not appreciate the communication sent may unsubscribe and it will not be possible to send them another campaign for which they could have been a good target

- The opposite is a missed opportunity, if no communication is sent to the most likely to purchase, a sale is missed. 
 
Without further information on the cost of losing a subscriber (false positive) or loss of missing a sale (false negative), it is still reasonable to say that the most important target to optimise is making sure to maximise the opportunities for sale.

There are limitations regarding the time at which the data was collected, when the purchase was made, the reason for missing values.

## DATA UNDERSTANDING

The transactions dataset contains 131337 observations of 17 variables: 
ID, Target (1 = purchase, 0 = no purchase),

age_band (85.5% null values)
![enter image description here](https://github.com/SandraLeriche/classification/blob/master/images/age_band_plot.png)

gender (52.77% null values) 
![enter image description here](https://github.com/SandraLeriche/classification/blob/master/images/gender_plot.png)
Vehicle type and history of maintenance: 
**Categorical** - car_model, car_segment 

![enter image description here](https://github.com/SandraLeriche/classification/blob/master/images/car_model_plot.png)

![enter image description here](https://github.com/SandraLeriche/classification/blob/master/images/car_segment_plot.png)

**Split into decile** - age_of_vehicle_years, sched_serv_warr, non_sched_serv_warr, sched_serv_paid, non_sched_serv_paid, total_paid_services, total_services, mth_since_last_serv, annualised_mileage, num_dealers_visited, num_serv_dealer_purchased

There is no further information on the high proportion of null values which can be related to the data collection processed and can be improved for future collection of data.

**Ethics and privacy considerations**
The amount of information available about the customer does not allow us to identify them (unless matched with another dataset). 

The team responsible for sending the campaign will have access to the personal data such as contact details. It is assumed that the data is collected in an ethical way with the customer's consent which may explain the lack of observations collected about the customer, and that they have opted in to receive communication. 

There may be a bias in gender since 52% of the data is missing and currently there are more male than female observations recorded, this bias could be further emphasized when applying resampling methods.

## DATA PREPARATION

Multicollinearity is showing in some variables which will be dropped when using logistic regression: sched_serv_paid, non_sched_serv_paid, sched_serv_warr, non_sched_serv_warr, total_paid_services
![enter image description here](https://github.com/SandraLeriche/classification/blob/master/images/multicollinearity.png)

There is an important imbalance in the dependent variable (Target).

It contains 97.3% of "No" and 2.7% of "Yes". Which can be corrected by using the SMOTE method which will oversample the "Yes" class and under-sample the "No" class to remove bias.

ID variable (not required) as well as the age_band and gender variables are excluded due to the high proportion of missing values.

The Target ID is converted to a categorical variable to allow for classification when training our model and making predictions.

## MODELLING

### Logistic Regression

This model assumes that there is a linear relationship between the logit of the outcome and each predictor variables and that there is no influential values (extreme values or outliers) in the continuous predictors.

The variables selected for the model after applying the stepwise selection (method that allows to select most influencial combination of variables) are:
car_model + age_of_vehicle_years + total_services + mth_since_last_serv + annualised_mileage + num_dealers_visited + num_serv_dealer_purchased

Confusion matrix result of the logistic regression model after applying the optimum cutoff: (see [Sensitivity / Specificity graph](https://github.com/SandraLeriche/classification/blob/master/images/sens_spec_log_reg.png) 

![enter image description here](https://github.com/SandraLeriche/classification/blob/master/images/cm_log_reg.JPG)                        
The ROC curve is available here: [ROC curve](https://github.com/SandraLeriche/classification/blob/master/images/roc_curve_log_reg.png)

THe model has an accuracy of 0.87 and sensitivity of 0.65 which is the metric that we are aiming to optimise, it represents the amount of true positive out of all the positives (TP/P).

### Decision Tree

All the variables are used for the decision tree model except for ID, gender and age_band.

Decision tree is not impacted by multicollinearity, it starts with the variable that can split the data into the biggest separation of Yes/No and reiterates until all observations have been split. 

By "pruning the tree", the tree becomes less complex and keeps the most significant variables. View the  [pruned decision tree](https://github.com/SandraLeriche/classification/blob/master/images/pruned_decision_tree.png).


The model has been trained on 70% of the data, the data has been split using partitioning in order to keep the balance of 97% of "0s" and 3% of "1s" in the Target variable.

Cross validation (5 repeats) was used to ensure all the patterns from the observations available in the training set are captured. 

SMOTE (oversampling approach) is used to correct the imbalance in the dataset to prevent biases of the decision being 0 and the final result being due to chance if the model is classifying most observations as "no". 

From there, the model is predicting the probability of getting a result of 1 (Yes) on the test set. 

To determine the threshold from which the probability will be converted into 1 or 0 we use the optimal cut-off using [sensitivity and specificity](https://github.com/SandraLeriche/classification/blob/master/images/sens_spec_dec_tree.png) which is 0.2 for the selected decision tree. 

The final model is selected by looking at a combination of different metrics:
the Area under the curver (AUC), confusion matrix results, accuracy score and sensitivity.

AUC: 0.85% for logistic regression and 0.94% for the decision tree - over 90% being considered as excellent.

Confusion matrix (predicted / actuals summary)

![enter image description here](https://github.com/SandraLeriche/classification/blob/master/images/cm_decision_tree.JPG)

Sensitivity: 0.79 for the regression model, 0.87 for the decision tree. 

Accuracy: 0.87 regression model vs 0.93 for the decision tree

The decision tree model is the one classifying more "1"s as "1" with a higher accuracy as well which is why this is the selected model.


## CONCLUSION

Regression is separating the no/yes values through a line whereas a decision tree is using the variables that can split as many observations as possible into two categories from highest to lowest. 

This explains the difference of variables being considered important for each model. 

For the regression model, the age of the vehicle, the total services, annualised mileage and number of services at the same dealer where the vehicle was purchased are highly impacting the decision of re-purchasing a vehicle. 

This means that a loyal customer who is servicing his vehicle at the same location is more likely to buy again once his vehicle hits a certain age or mileage.

For the decision tree, the important variables are mostly related to the services (months since the last service, check ups used under warranty, scheduled and paid services, total_services and the age of the vehicle regardless of the location of the services, but still using official dealerships.
It also shows that loyalty to the brand (not specifically the local shop) and the image they have of the brand and services received is taken into consideration when the customer decides to buy another vehicle.

The most Important variables for the business to collect and used for this specific model are "mth_since_last_serv" "sched_serv_warr" "sched_serv_paid" "total_services" "age_of_vehicle_years". 

See how the variables impact the Target (dependent) variable by looking at the [partial dependency plots](https://github.com/SandraLeriche/classification/tree/master/images/partial_dependency_plots).

Total_services is standing out as it is flat, which means it is a predictor of our Target variable by 0.10 as soon as there is 1 service done it's impact does not change the more services the customer has done.

**Next steps:**
Further changes can be made to the model should the information about the customer (age, gender) become more significant and by collecting additional data to train the model.