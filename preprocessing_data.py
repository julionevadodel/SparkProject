import pyspark.sql.functions as f
from pyspark.ml.feature import StringIndexer, UnivariateFeatureSelector, VectorAssembler, OneHotEncoder
from pyspark.ml import Pipeline

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator



#########################################   CLEAN DATA     ##############################
def clean_data(df):
    df = df.drop('ArrTime','ActualElapsedTime','AirTime','TaxiIn','Diverted','CarrierDelay',
                'WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay','FlightNum')


    df = df.withColumn("DepHour",f.regexp_replace(f.col("DepTime"),"(\\d{1,2})(\\d{2})","$1"))
    df = df.withColumn("DepMinute",f.regexp_replace(f.col("DepTime"),"(\\d{1,2})(\\d{2})","$2"))
    df = df.withColumn("CRSDepHour",f.regexp_replace(f.col("CRSDepTime"),"(\\d{1,2})(\\d{2})","$1"))
    df = df.withColumn("CRSDepMinute",f.regexp_replace(f.col("CRSDepTime"),"(\\d{1,2})(\\d{2})","$2"))
    df = df.withColumn("CRSArrHour",f.regexp_replace(f.col("CRSArrTime"),"(\\d{1,2})(\\d{2})","$1"))
    df = df.withColumn("CRSArrMinute",f.regexp_replace(f.col("CRSArrTime"),"(\\d{1,2})(\\d{2})","$2"))
    #df = df.withColumn("DepTime",f.regexp_replace(f.col("DepTime"),"(\\d{1,2})(\\d{2})","$1:$2"))
    #df = df.withColumn("CRSDepTime",f.regexp_replace(f.col("CRSDepTime"),"(\\d{1,2})(\\d{2})","$1:$2"))
    #df = df.withColumn("CRSArrTime",f.regexp_replace(f.col("CRSArrTime"),"(\\d{1,2})(\\d{2})","$1:$2"))
    #df = df.withColumn("Date",f.concat_ws("-","Year","Month","DayofMonth"))
    #df = df.drop("Year","Month","DayofMonth")
    df = df.drop("Year")
   
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
    df = df.select(f.col("DayOfWeek").cast('int').alias("DayOfWeek"),#CATEGORICAL
                    f.col("CRSElapsedTime").cast('double').alias("CRSElapsedTime"),#NUMERICAL
                    f.col("DepDelay").cast('double').alias("DepDelay"),#NUMERICAL
                    f.col("Distance").cast('double').alias("Distance"),#NUMERICAL
                    f.col("TaxiOut").cast('double').alias("TaxiOut"),#NUMERICAL
                    f.col("DepHour").cast('int').alias("DepHour"),#NUMERICAL DISCRETE
                    f.col("DepMinute").cast('int').alias("DepMinute"), #NUMERICAL DISCRETE
                    f.col("CRSDepHour").cast('int').alias("CRSDepHour"),#NUMERICAL DISCRETE
                    f.col("CRSDepMinute").cast('int').alias("CRSDepMinute"),#NUMERICAL DISCRETE
                    f.col("CRSArrHour").cast('int').alias("CRSArrHour"),#NUMERICAL DISCRETE
                    f.col("CRSArrMinute").cast('int').alias("CRSArrMinute"),#NUMERICAL DISCRETE
                    f.col("Month").cast('int').alias("Month"),#CATEGORICAL
                    f.col("DayOfMonth").cast('int').alias("DayOfMonth"),#NUMERICAL
                    #f.col("ArrDelay").cast('double').alias("ArrDelay"),'Date','DepTime','CRSDepTime','CRSArrTime','UniqueCarrier',
                    f.col("ArrDelay").cast('double').alias("ArrDelay"),#NUMERICAL
                    'UniqueCarrier',#NOMINAL
                    'TailNum',#NOMINAL
                    'Origin',#NOMINAL
                    'Dest') #NOMINAL
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
    columns = ["DayOfWeek","Month","UniqueCarrier","TailNum","Origin","Dest"]
    indexers = [StringIndexer(inputCol=col,outputCol=col+"_index").setHandleInvalid("skip") for col in columns] #Error with some TailNum since we fit on train and, if not present in that set and then appears in train our fitted pipeline does not know what to do

    ohe_input_cols = [col+"_index" for col in columns]
    ohe_output_cols = [col+"_vect" for col in columns]
    ohe = OneHotEncoder(inputCols=ohe_input_cols,outputCols=ohe_output_cols)

    return indexers+[ohe]



#########################################   SPLIT DATA   ##############################
def split_data(df):
    df_train,df_test = df.([0.7,0.3],24)

    return df_train,df_test



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
    columns = ["DayOfWeek_vect","CRSElapsedTime","DepDelay","Distance","TaxiOut","DayOfMonth","DepHour","DepMinute","CRSDepHour","CRSDepMinute",
                "CRSArrHour","CRSArrMinute","Month_vect","UniqueCarrier_vect","TailNum_vect","Origin_vect","Dest_vect"]
    assembler = VectorAssembler(inputCols=columns,outputCol="Features")
    ufss = UnivariateFeatureSelector(featuresCol="Features",outputCol="selectedFeatures",
                                    labelCol="ArrDelay",selectionMode="fpr")
    ufss.setFeatureType("continuous").setLabelType("continuous").setSelectionThreshold(0.05)
    return [assembler,ufss]
    




#########################################   PIPELINE  ##############################
def create_pipeline(stages):
    pipeline=Pipeline().setStages(stages)
    return pipeline

def fit_pipeline(pipeline,df):
    pipeline = pipeline.fit(df)

    #for i in df.dtypes:
    #    if i[1] == "string":
    #        df = df.drop(i[0])
    #        df = df.withColumnRenamed(i[0]+"_index",i[0])
        
    return pipeline

def apply_pipeline(pipeline,df):
    results = pipeline.transform(df)
    return results


#########################################   VectorAssembler  ##############################    
def create_vectorAssem(df):
    vectorAssembler_str = VectorAssembler(inputCols=['DayOfWeek','CRSElapsedTime','DepDelay','Distance','TaxiOut','DepTime_index','CRSDepTime_index','CRSArrTime_index','Origin_index','Dest_index'], 
    outputCol="features", handleInvalid="skip")

    df= vectorAssembler_str.transform(df_string_converted)

    return df 

def create_final_set(df):
    df=df.select('features','ArrDelay')

    return df




#########################################   Linear Regression  ##############################
def apply_linear_regression(df_train,df_test):
    # creates regressor
    regressor = LinearRegression(featuresCol = 'features', labelCol = 'ArrDelay')

    # trains model
    regressor = regressor.fit(df_train)
    print_metrics_LR(regressor)

    # test/evaluate model
    evaluator=regressor.evaluate(df_test)
    predictions= evaluator.predictions

  return predictions


def print_metrics_LR(regressor):
    
    print("Linear Regression")

    # What is the relationship between my dependent (x) and independent (y) variable? 
    print("Coefficients: %s" % regressor.coefficients)
    
    # How much of the data does this model explain?
    print("R-squared: %f" % regressor.summary.r2)    
    
    # square root of the average of the squared difference of the predicted and actual value
    print("RMSE Root Mean Square Error: %f" % regressor.summary.rootMeanSquaredError)
    

#########################################   Cross validation  ##############################
def crossValidation(df):
    regression = LinearRegression(labelCol='ArrDelay')
    evaluator = RegressionEvaluator(labelCol='ArrDelay')

    # creates cross validator with 5 folds
    cv = CrossValidator(estimator=regression, estimatorParamMaps=params, evaluator=evaluator, numFolds=5)

    # Train and test model on 5 different folds="sets" of data
    cv = cv.fit(df)
    print_metrics_LR(cv)
    bestModel=cv.bestModel

    return bestModel 

 
 def print_metrics_LR(cv):
    
    print("Linear Regression. Best Model ")

    # What is the relationship between my dependent (x) and independent (y) variable? 
    print("Coefficients: %s" % cv.bestModel.coefficients)
    
    # How much of the data does this model explain?
    print("R-squared: %f" % cv.bestModel.summary.r2)    
    
    # square root of the average of the squared difference of the predicted and actual value
    print("RMSE: %f" % regressor.summary.rootMeanSquaredError)   
