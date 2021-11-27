import pyspark.sql.functions as f
from pyspark.ml.feature import StringIndexer, UnivariateFeatureSelector, VectorAssembler
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


def drop_null(df):
    df= df.na.drop()
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
    print("\nDifferent dates: ",df.select("Date").distinct().collect())
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
    return indexers




#########################################   FEATURE SELECTION  ##############################
#def feature_subset_selection(df):

    cat_mask = [var_type=='string' for (var_name,var_type) in df.dtypes]
    num_mask = [var_type=='double' for (var_name,var_type) in df.dtypes]

    cat_cols = [df.columns[i] for i in range(len(df.columns)) if cat_mask[i]]
    num_cols = [df.columns[i] for i in range(len(df.columns)) if num_mask[i]]

    ufss_cat = [UnivariateFeatureSelector(featuresCol=col,outputCol=col+"_selected",
                                    labelCol="ArrDelay",selectionMode='fpr')\
                                    .setSelectionThreshold(0.5)\
                                    .setFeatureType("categorical")\
                                    .setLabelType("continuous").fit(df)    
                                    for col in cat_cols]
    ufss_num = [UnivariateFeatureSelector(featuresCol=col,outputCol=col+"_selected",
                                    labelCol="ArrDelay",selectionMode='fpr')\
                                    .setSelectionThreshold(0.5)\
                                    .setFeatureType("continuous")\
                                    .setLabelType("continuous").fit(df) 
                                     for col in num_cols]
    ufss = ufss_cat+ufss_num
    return ufss
def feature_subset_selection(df):
    columns = []
    for i in df.dtypes:
        if i[1] == 'double':
            columns.append(i[0])
        else:
            columns.append(i[0]+"_index")
    columns.remove("ArrDelay")
    assembler = VectorAssembler(inputCols=columns,outputCol="Features")
    ufss = UnivariateFeatureSelector(featuresCol="Features",outputCol="selectedFeatures",
                                    labelCol="ArrDelay",selectionMode="fpr")
    ufss.setFeatureType("continuous").setLabelType("continuous").setSelectionThreshold(0.05)
    return [assembler,ufss]
    
    




#########################################   PIPELINE  ##############################
def create_pipeline(stages):
    pipeline=Pipeline().setStages(stages)
    return pipeline

def apply_pipeline(pipeline,df):
    df = pipeline.transform(df)

    for i in df.dtypes:
        if i[1] == "string":
            df = df.drop(i[0])
            df = df.withColumnRenamed(i[0]+"_index",i[0])
        
    return df

