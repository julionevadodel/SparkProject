# Loading the data

# Process the data

# Creating the model


## Linear Regression
Linear regression model was chosen for the presents set because..

After the loading and processing of data steps described above, the vector with following features was created. DayOfWeek','CRSElapsedTime','DepDelay','Distance','TaxiOut','DepTime_index','CRSDepTime_index','CRSArrTime_index','Origin_inde'','Dest_index'. 

To train the model the regressor takes in the input of features (the vector decribed above) and the name of the colun that represents the feature to be predicted, in this case 'ArrDelay'. The training is performed on train_data which includes 7/10 of data. The metrics metricsof the model found after performing the training are presented in the screenshot below.

--pic--

 - R-squared measures the strength of the relationship between the input vector of features (features) and the dependent variable(ArrDelay). In other words it answers the question: "How much of the data does this model explain? ". The value of 0.939473 means that 93% of the data represent the variance of the dependent variable (ArrDelay). 
 - Coefficience decribes relationship between the dependent variable (x /vector) and independent (y/ArrDelay) variable. Results presented below show that the attributes with the most significant influence (0ver 0.9) on ArrDelay are DepDelay are TaxiOut. The strength of relationship between CRSElapsedTime and our independent varible is 0.2. 

- pic- 


 - Root Mean Square Error (RMSE) is the square root of the average of the squared difference of the predicted and actual value. It is used as ameasure of accuracy, to compare prediction errors of different models. The model with the lowest RMSE is the best one.
 

Following, the model was applied to the test_data set. The prediction calculated for top 10 raws in the data are showen bellow.

--pic--
--explain--


# Validating the model

Metrics Linear Regression:
- The *coefficients* is a list, where the first element indicates how flight delay changes by chosen attributes
- The *intercept* is a ..
- Loss function = MSE(Mean Square Error) = (observed values - model values)^2
- RegresionEvaluator: mae(mean absolute error), r2 


## Linear Regression
Linear regression aims to minimize the MSE


## CrossValidation for Linear Regression
CrossValidation allows us to compare different machine learning methods and get a sence of how well they will work in practice. 

Rather the spliting the data into train and test sets crossvalidation separates data into the folds or "blocks" and uses one at the time,and summarizes the results obtained in the end. 

For the performing cross validation for linear Regression the amount of 5 folds was chosen. The metrics of the best model found after performing the validation are presented in the screenshot below.

--pic--



# Advanced creterias fullfilled
- 3 machine learning algorithm
- CrossValidation 
- Proper exploratory data analysis 
- Smart use of the Spark tools
- Smart handling of special format 
