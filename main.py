from pyspark.sql import SparkSession
import os
import sys
from preprocessing_data import *
from modelling_data import *   
from pyspark.sql.functions import rand

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

def construct_pipeline(df):
    indexers = cat_to_num(df)
    ufss = feature_subset_selection(df)
    DT = decision_tree(df)
    evaluator = evaluate_model()
    cv = cross_validate(DT,evaluator)
    stages = indexers
    stages = stages+ufss+cv
    pipeline = create_pipeline(stages)
    return pipeline,evaluator




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
        #df = spark.sparkContext.parallelize(df.orderBy(rand()).take(1000)).toDF()
        df_tr,df_tst = preprocess_data(df)
        pipeline,evaluator = construct_pipeline(df_tr)

        pipeline = fit_pipeline(pipeline,df_tr)
        results = apply_pipeline(pipeline,df_tst)
        results.show(5)
        #print(results.dtypes)
        #for result in results.head(3):
        #    print(result.Features,"\t // \t",result.selectedFeatures)
        #show_missing_values(results) 
        print(evaluator.evaluate(results))


    
    

 
    
    
   




####################################### TESTS ########################################## 
path = "2008.csv"
#path="oficinas_farmacia.tsv"
main(path)
    