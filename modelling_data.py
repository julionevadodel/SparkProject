from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder



########################################## CV ######################
def cross_validate(estimator,evaluator):
    paramGrid = ParamGridBuilder() \
                .baseOn({estimator.featuresCol : 'selectedFeatures'}) \
                .baseOn({estimator.labelCol : 'ArrDelay' })\
                .baseOn({estimator.predictionCol : 'prediction' })\
                .addGrid(estimator.maxDepth, range(5,16)) \
                .build()
    cv = CrossValidator(estimator = estimator, evaluator = evaluator, estimatorParamMaps=paramGrid)
    return [cv]






########################################## Evaluate ######################
def evaluate_model():
     evaluator = RegressionEvaluator(predictionCol='prediction',labelCol='ArrDelay',metricName='rmse')
     return evaluator



########################################## DecisionTreeRegressor ######################
def decision_tree(df):
    DT = DecisionTreeRegressor(featuresCol="selectedFeatures",labelCol="ArrDelay",predictionCol="prediction")
    return DT

