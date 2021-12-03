from pyspark.sql import SparkSession
import os
import sys
from preprocessing_data import *
from modelling_data import *   
from pyspark.sql.functions import rand
from pyspark.ml.tuning import ParamGridBuilder

def read_file(spark,path):
    try:
        df = spark.read.csv(path,header=True)
    except:
        return "Path not valid"
    try:   
        if df.count()==0:
            return "File is empty"
        expected_columns = ["Year","Month","DayofMonth","DayOfWeek","DepTime","CRSDepTime",
                            "ArrTime","CRSArrTime","UniqueCarrier","FlightNum","TailNum",
                            "ActualElapsedTime","CRSElapsedTime","AirTime","ArrDelay",
                            "DepDelay","Origin","Dest","Distance","TaxiIn","TaxiOut","Cancelled",
                            "CancellationCode","Diverted","CarrierDelay","WeatherDelay","NASDelay",
                            "SecurityDelay","LateAircraftDelay"]
        obtained_columns = df.columns
        for col in obtained_columns:
            if col not in expected_columns:
                return "Column " + col + " is missing"
        return df
    except:
        return "Something went wrong"



def preprocess_data(df):
    df = clean_data(df)
    df = cast_data(df)
    df = clean_data2(df)

    df_tr,df_tst = df.randomSplit([0.7,0.3],24)


    return df_tr,df_tst

def construct_preprocessing_pipeline(df):
    indexers = cat_to_num(df)
    ufss = feature_subset_selection(df)
    stages = indexers
    stages = stages+ufss
    pipeline = create_pipeline(stages)
    return pipeline

def construct_tree_pipeline():
    DT = decision_tree()
    evaluator = evaluate_model()
    paramGrid = ParamGridBuilder() \
                .baseOn({DT.featuresCol : 'selectedFeatures'}) \
                .baseOn({DT.labelCol : 'ArrDelay' })\
                .baseOn({DT.predictionCol : 'prediction' })\
                .addGrid(DT.maxDepth, [5, 7, 11, 13, 15]) \
                .build()
    cv = cross_validate(DT,evaluator,paramGrid)
    pipeline = create_pipeline(cv)
    return pipeline,evaluator

def construct_lr_pipeline():
    LR = linear_regression()
    evaluator = evaluate_model()
    paramGrid = ParamGridBuilder() \
                .baseOn({LR.featuresCol : 'selectedFeatures'}) \
                .baseOn({LR.labelCol : 'ArrDelay' })\
                .baseOn({LR.predictionCol : 'prediction' })\
                .build()
    cv = cross_validate(LR,evaluator,paramGrid)
    pipeline = create_pipeline(cv)
    return pipeline,evaluator

def construct_glr_pipeline():
   GLR = generalized_linear_regression()
   evaluator = evaluate_model()
   paramGrid = ParamGridBuilder() \
               .baseOn({GLR.featuresCol : 'selectedFeatures'}) \
               .baseOn({GLR.labelCol : 'ArrDelay' })\
               .baseOn({GLR.predictionCol : 'prediction' })\
               .build()
   cv = cross_validate(GLR, evaluator, paramGrid)
   pipeline = create_pipeline(cv)
   return pipeline, evaluator

#def construct_vector(df):
#    df=create_vectorAssem(df)
#    df=create_final_set(df)

#    return df

#def data_preparation_LR (df):  
#    df = clean_data(df)
#    df = cast_data(df)
#    df = clean_data2(df)
#    df=drop_null(df)
#    df=organise_data(df)
#    indexers = cat_to_num(df)
#    df= construct_pipeline(indexers).transform(df)  # include transform to construct_pipeline- method
#    df=construct_vector(df)

#    return df



#def linearRegression(df):
#    df_train,df_test=split_data(df)
#    apply_linear_regression(df_train,df_test)







def main(path):
    os.environ['PYSPARK_PYTHON'] = sys.executable           #CHECK
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = SparkSession.builder\
                        .appName("Big Data & Data Visualization: Assignment 1")\
                        .master("local")\
                        .enableHiveSupport()\
                        .getOrCreate()

    df = read_file(spark,path)
    if type(df) == str:
        print(df)
    else:
        df = spark.sparkContext.parallelize(df.orderBy(rand()).take(1000)).toDF()
        df_tr,df_tst = preprocess_data(df)
        
        pipeline_preprocessing = construct_preprocessing_pipeline(df_tr)

        pipeline_preprocessing = fit_pipeline(pipeline_preprocessing,df_tr)
        df_tr_prec = apply_pipeline(pipeline_preprocessing,df_tr).cache()
        df_tst_prec = apply_pipeline(pipeline_preprocessing,df_tst).cache()


        


        #DecisionTreeRegressor
        tree_pipeline,evaluator = construct_tree_pipeline()
        tree_pipeline = fit_pipeline(tree_pipeline,df_tr_prec)
        results_tree = apply_pipeline(tree_pipeline,df_tst_prec)

        results_tree.show(5)

        print(evaluator.evaluate(results_tree))


    
    
        # linear regression
        lr_pipeline,evaluator = construct_lr_pipeline()
        lr_pipeline = fit_pipeline(lr_pipeline,df_tr_prec)
        results_lr = apply_pipeline(lr_pipeline,df_tst_prec)
        
        results_lr.show(5)

        print(evaluator.evaluate(results_lr))

        #  linear regression best model
        #bestModel=crossValidation(df)

        # Generalized Linear Regression
        glr_pipeline, evaluator = construct_glr_pipeline()
        glr_pipeline = fit_pipeline(glr_pipeline, df_tr_prec)
        results_glr = apply_pipeline(glr_pipeline, df_tst_prec)
                    
        results_glr.show(5)

        print(evaluator.evaluate(results_glr))
        print(glr_pipeline.stages[0].getEstimator())
        prints_metrics_GLR(model = glr_pipeline.stages[0].bestModel)
 
####################################### TESTS ########################################## 
path = "2008.csv"
#path="oficinas_farmacia.tsv"
main(path)

