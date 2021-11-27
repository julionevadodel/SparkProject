from pyspark.sql import SparkSession
import os
import sys
from preprocessing_data import *


def preprocess_data(df):
    df = clean_data(df)
    df = cast_data(df)
    df = clean_data2(df)

    df_tr,df_tst = df.randomSplit([0.7,0.3],24)


    return df_tr,df_tst

def construct_pipeline(df):
    indexers = cat_to_num(df)
    ufss = feature_subset_selection(df)
    stages = indexers
    stages = stages+ufss
    pipeline = create_pipeline(stages)
    pipeline = pipeline.fit(df)
    return pipeline

def construct_vector(df):
    df=create_vectorAssem(df)
    df=create_final_set(df)

    return df

def data_preparation_LR (df):  
    df = clean_data(df)
    df = cast_data(df)
    df = clean_data2(df)
    df=drop_null(df)
    df=organise_data(df)
    indexers = cat_to_num(df)
    df= construct_pipeline(indexers).transform(df)  # include transform to construct_pipeline- method
    df=construct_vector(df)

    return df



def linearRegression(df):
    df_train,df_test=split_data(df)
    apply_linear_regression(df_train,df_test)







def main(path):
    os.environ['PYSPARK_PYTHON'] = sys.executable           #CHECK
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = SparkSession.builder\
                        .appName("Big Data & Data Visualization: Assignment 1")\
                        .master("local")\
                        .enableHiveSupport()\
                        .getOrCreate()

    df = spark.read.csv(path,header=True)
    
    df_tr,df_tst = preprocess_data(df)
    pipeline = construct_pipeline(df_tr)

    df_tst_transformed = apply_pipeline(pipeline,df_tst)

    df_tst_transformed.show(2)
    
    
# linear regression
    df_lr=data_preparation_LR()
    
    # linear regression
    predictions=linearRegression(df_lr)
    predictions.show(10)

    #  linear regression best model
    bestModel=crossValidation(df)

 
    
    
   




####################################### TESTS ########################################## 
path = "2008.csv"
main(path)
