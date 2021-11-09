from pyspark.sql import SparkSession
import os
import sys
from preprocessing_data import *






def main(path):
    os.environ['PYSPARK_PYTHON'] = sys.executable           #CHECK
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = SparkSession.builder\
                        .appName("Big Data & Data Visualization: Assignment 1")\
                        .master("local")\
                        .enableHiveSupport()\
                        .getOrCreate()

    df = spark.read.csv(path,header=True)

    df = clean_data(df)
    show_missing_values(df)

    df = cast_data(df)
    show_missing_values(df)

    df = clean_data2(df)
    show_missing_values(df)

    df = cat_to_num(df)
    show_missing_values(df)

    #corr = var_correlations(df)
    #print(df.dtypes)

    df.show(2)
    
    

 
    
    
   




####################################### TESTS ########################################## 
path = "2008.csv"
main(path)
    