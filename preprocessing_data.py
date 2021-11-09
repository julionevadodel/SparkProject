import pyspark.sql.functions as f
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline




#########################################   CLEAN DATA     ##############################
def clean_data(df):
    df = df.drop('ArrTime','ActualElapsedTime','AirTime','TaxiIn','Diverted','CarrierDelay',
                'WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay','FlightNum')


    df = df.withColumn("DepTime",f.regexp_replace(f.col("DepTime"),"(\\d{1,2})(\\d{2})","$1:$2"))
    df = df.withColumn("CRSDepTime",f.regexp_replace(f.col("CRSDepTime"),"(\\d{1,2})(\\d{2})","$1:$2"))
    df = df.withColumn("CRSArrTime",f.regexp_replace(f.col("CRSArrTime"),"(\\d{1,2})(\\d{2})","$1:$2"))
    df = df.withColumn("Date",f.concat_ws("-","Year","Month","DayofMonth"))
    df = df.drop("Year","Month","DayofMonth")
   
    return df


def clean_data2(df):
    df = df.dropna(subset=["ArrDelay","TailNum"]) #If no flight, useless for ArrDelay?
    df = df.drop("Cancelled","CancellationCode")
    return df





#########################################   ORGANISE DATA   ##############################
def organise_data(df):
    df = df[["Date","DayOfWeek","DepTime","CRSDepTime","CRSArrTime","UniqueCarrier","TailNum","CRSElapsedTime",\
        "DepDelay","Origin","Dest","Distance","TaxiOut","ArrDelay"]] #Add Cancelled and Cancellation code if necessary
    return df
    




#########################################   CAST DATA   ##############################
def cast_data(df):
    df = df.select(f.col("DayOfWeek").cast('double').alias("DayOfWeek"),
                    f.col("CRSElapsedTime").cast('double').alias("CRSElapsedTime"),
                    f.col("DepDelay").cast('double').alias("DepDelay"),
                    f.col("Distance").cast('double').alias("Distance"),
                    f.col("TaxiOut").cast('double').alias("TaxiOut"),
                    f.col("ArrDelay").cast('double').alias("ArrDelay"),'Date','DepTime','CRSDepTime','CRSArrTime','UniqueCarrier',
                    'TailNum','Origin','Dest')
    #df = df.withColumn("DayOfWeek",df["DayOfWeek"].cast('int'))
    #df = df.withColumn("CRSElapsedTime",df["CRSElapsedTime"].cast('int'))
    #df = df.withColumn("DepDelay",df["DepDelay"].cast('int'))
    #df = df.withColumn("Distance",df["Distance"].cast('int'))
    #df = df.withColumn("TaxiOut",df["TaxiOut"].cast('int'))
    #df = df.withColumn("ArrDelay",df["ArrDelay"].cast('int'))
    #df = df.withColumn("Cancelled",df["Cancelled"].cast('integer'))

    return df




#########################################   SHOW DISTINCT VALUES   ##############################
def show_distinct_values(df):
    print("\nDifferent carriers: ",df.select("UniqueCarrier").distinct().count())
    print("\nDifferent TailNum: ",df.select("TailNum").distinct().count())
    print("\nDifferent Origin: ",df.select("Origin").distinct().count())
    print("\nDifferent Destination: ",df.select("dest").distinct().count())





#########################################   SHOW MISSING VALUES   ##############################
def show_missing_values(df):
    df.select([f.count(f.when(f.col(c).isNull(), c)).alias(c) for c in df.columns]).show()




#########################################   COMPUTE CORRELATIONS   ##############################
def var_correlations(df):
    num_mask = [var_type=='double' for (var_name,var_type) in df.dtypes]
    num_cols = [df.columns[i] for i in range(len(df.columns)) if num_mask[i]]
    df_num = df[num_cols]    
    corr = [[df_num.corr(c1,c2) for c1 in df_num.columns] for c2 in df_num.columns]
    print(num_cols)
    [print("\n",corr_row) for corr_row in corr]
    return corr




#########################################   CAT TO NUM   ##############################
def cat_to_num(df):
    cat_mask = [var_type=='string' for (var_name,var_type) in df.dtypes]
    cat_cols = [df.columns[i] for i in range(len(df.columns)) if cat_mask[i]]
    df_cat = df[cat_cols]
    columns = df_cat.columns

    indexers = [StringIndexer(inputCol=col,outputCol=col+"_index").fit(df_cat) for col in columns]
    pipeline=Pipeline(stages=indexers)
    df_r = pipeline.fit(df).transform(df)

    for col in columns:
        df_r = df_r.drop(col)
        df_r = df_r.withColumnRenamed(col+"_index",col)

    df_r = organise_data(df_r)
    

    return df_r
