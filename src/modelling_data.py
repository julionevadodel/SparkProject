from pyspark.ml.regression import DecisionTreeRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder



########################################## CV ######################
def cross_validate(estimator,evaluator, paramGrid):
    cv = CrossValidator(estimator = estimator, evaluator = evaluator, estimatorParamMaps=paramGrid)
    return [cv]






########################################## Evaluate ######################
def evaluate_model():
     evaluator = RegressionEvaluator(predictionCol='prediction',labelCol='ArrDelay',metricName='rmse')
     return evaluator



########################################## DecisionTreeRegressor ######################
def decision_tree():
    DT = DecisionTreeRegressor(featuresCol="selectedFeatures",labelCol="ArrDelay",predictionCol="prediction")
    return DT



########################################## LinearRegression ######################
def linear_regression():
    LR = LinearRegression(featuresCol = 'features', labelCol = 'ArrDelay')
