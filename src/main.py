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
        return "Path not valid" #If an exception is thrown the user is told that the provided path to file is not valid
    try:   
        if df.count()==0: #If there are 0 columns then the file is empty
            return "File is empty"
        expected_columns = ["Year","Month","DayofMonth","DayOfWeek","DepTime","CRSDepTime",
                            "ArrTime","CRSArrTime","UniqueCarrier","FlightNum","TailNum",
                            "ActualElapsedTime","CRSElapsedTime","AirTime","ArrDelay",
                            "DepDelay","Origin","Dest","Distance","TaxiIn","TaxiOut","Cancelled",
                            "CancellationCode","Diverted","CarrierDelay","WeatherDelay","NASDelay",
                            "SecurityDelay","LateAircraftDelay"] #We admit only these columns in order to assess the correct performance of the program
        obtained_columns = df.columns
        for col in obtained_columns:
            if col not in expected_columns:
                return "Column " + col + " is missing" #Missing columns are alerted
        return df
    except:
        return "Something went wrong" #If an error ocurred and none of the previous cases is fulfilled



def preprocess_data(df): #Initial data preprocessing
    df = clean_data(df) #Forbidden variables are removed. Time variables are splitted into hour and minute
    df = cast_data(df) #Variables are converted to suitable datatypes
    df = clean_data2(df) #Deal with missing values

    df_tr,df_tst = df.randomSplit([0.7,0.3],24) #Train is 70pct of the df and test the 30pct
    return df_tr,df_tst



def construct_preprocessing_pipeline(df):
    indexers = cat_to_num(df) #Conversion of categorical variables to numeric. First StringIndexer. Second OHE
    ufss = feature_subset_selection(df) #univariate feature subset selection. Only the most important remain. First Vector Assembler. Then UFSS.
    stages = indexers
    stages = stages+ufss #Creation of a list of stages
    pipeline = create_pipeline(stages) #Pipeline is created by passing the pipeline stages a list of stages.
    return pipeline


def construct_tree_pipeline():
    DT = decision_tree() #DecisionTreeRegressor is obtained.
    evaluator = evaluate_model() #Evaluator is obtained. RMSE is the metric used.
    paramGrid = ParamGridBuilder() \
                .baseOn({DT.featuresCol : 'selectedFeatures'}) \
                .baseOn({DT.labelCol : 'ArrDelay' })\
                .baseOn({DT.predictionCol : 'prediction' })\
                .addGrid(DT.maxDepth, [5, 7, 11, 13, 15]) \
                .build() #A param grid to cross validate different maximum depth for the Tree
    cv = cross_validate(DT,evaluator,paramGrid) #The cross validation element is created.
    pipeline = create_pipeline(cv) #Modelling pipeline is created by setting its only stage.
    return pipeline,evaluator


def construct_lr_pipeline():
    LR = linear_regression() #Linear Regressor object is obtained
    evaluator = evaluate_model() #Evaluator is obtained. RMSE is the metric used.
    paramGrid = ParamGridBuilder() \
                .baseOn({LR.featuresCol : 'selectedFeatures'}) \
                .baseOn({LR.labelCol : 'ArrDelay' })\
                .baseOn({LR.predictionCol : 'prediction' })\
                .build()
    cv = cross_validate(LR,evaluator,paramGrid) #The cross validation element is created. In order to check several train - val splits to ensure the performance of the model
    pipeline = create_pipeline(cv) #Modelling pipeline is created by setting its only stage.
    return pipeline,evaluator


def construct_glr_pipeline():
   GLR = generalized_linear_regression() #Generalized Linear Regressor object is obtained
   evaluator = evaluate_model() #Evaluator is obtained. RMSE is the metric used.
   paramGrid = ParamGridBuilder() \
               .baseOn({GLR.featuresCol : 'selectedFeatures'}) \
               .baseOn({GLR.labelCol : 'ArrDelay' })\
               .baseOn({GLR.predictionCol : 'prediction' })\
               .build()
   cv = cross_validate(GLR, evaluator, paramGrid) #The cross validation element is created. In order to check several train - val splits to ensure the performance of the model
   pipeline = create_pipeline(cv) #Modelling pipeline is created by setting its only stage.
   return pipeline, evaluator








def main(path):
    os.environ['PYSPARK_PYTHON'] = sys.executable           #CHECK
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = SparkSession.builder\
                        .appName("Big Data & Data Visualization: Assignment 1")\
                        .master("local")\
                        .enableHiveSupport()\
                        .getOrCreate() #Spark session builder in case user does not externally configure spark context.

    df = read_file(spark,path) #Checking input file
    if type(df) == str: #If file is not suitable a string with an error is returned
        print(df)
    else: #If file is valid the dataframe containing the observations is returned
        df = spark.sparkContext.parallelize(df.orderBy(rand()).take(1000)).toDF() #Randomly select 1000 observations. Test purposes only
        df_tr,df_tst = preprocess_data(df) #Initial call to preprocessing. Variables are casted. Missing values ruled out. Forbidden varibles eliminated
        
        pipeline_preprocessing = construct_preprocessing_pipeline(df_tr) #Deeper preprocessing pipeline is created.

        pipeline_preprocessing = fit_pipeline(pipeline_preprocessing,df_tr) #Preprocessing pipelien is fit on training set
        df_tr_prec = apply_pipeline(pipeline_preprocessing,df_tr).cache() #Created pipelines is used to transform test and training sets
        df_tst_prec = apply_pipeline(pipeline_preprocessing,df_tst).cache()
        #Both dfs are cached since they will be used in three different models


        


        ################################# Decision Tree Rregressor #################################
        tree_pipeline,evaluator = construct_tree_pipeline() #Create the pipeline for DecisionTreeRregressor
        tree_pipeline = fit_pipeline(tree_pipeline,df_tr_prec) #Fit previous pipeline on train data
        results_tree = apply_pipeline(tree_pipeline,df_tst_prec) #Apply the previously fitted pipeline on test data

        results_tree.show(5)

        print(evaluator.evaluate(results_tree)) #RMSE obtained for test set is shown
        
        #Decision tree regressor best model
        print_metrics_DT(tree_pipeline.stages[0].bestModel)


    
    
        ################################# Linear Regressor #################################
        lr_pipeline,evaluator = construct_lr_pipeline() #Pipeline for linear regresion is created
        lr_pipeline = fit_pipeline(lr_pipeline,df_tr_prec) #Fit previous pipeline on train data
        results_lr = apply_pipeline(lr_pipeline,df_tst_prec) #Apply the previously fitted pipeline on test data
        
        results_lr.show(5)

        print(evaluator.evaluate(results_lr)) #RMSE obtained for test set is shown

        #Linear regression best model
        print_metrics_LR(lr_pipeline.stages[0].bestModel)



        ################################# Generalized Linear Regressor #################################
        glr_pipeline, evaluator = construct_glr_pipeline() #Pipeline for generalized linear regression is created
        glr_pipeline = fit_pipeline(glr_pipeline, df_tr_prec) #Fit previous pipeline on train data
        results_glr = apply_pipeline(glr_pipeline, df_tst_prec) #Apply the previously fitted pipeline on test data
                    
        results_glr.show(5)

        print(evaluator.evaluate(results_glr)) #RMSE for test set is shown

        #Generalized linear regression best model
        prints_metrics_GLR(glr_pipeline.stages[0].bestModel)
 





####################################### INITIAL CALL ########################################## 
path = "2008.csv"
#path="oficinas_farmacia.tsv"
main(path)

