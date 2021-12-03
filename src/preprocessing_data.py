import pyspark.sql.functions as f
from pyspark.ml.feature import StringIndexer, UnivariateFeatureSelector, VectorAssembler, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator



#########################################   CLEAN DATA     ##############################
def clean_data(df):
    #Forbidden variables are dropped as well as FlightNum
    df = df.drop('ArrTime','ActualElapsedTime','AirTime','TaxiIn','Diverted','CarrierDelay',
                'WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay','FlightNum')

    #Time variables containing hours and minutes are splitted into two columns each. One for hours. One for minutes.
    df = df.withColumn("DepHour",f.regexp_replace(f.col("DepTime"),"(\\d{1,2})(\\d{2})","$1"))
    df = df.withColumn("DepMinute",f.regexp_replace(f.col("DepTime"),"(\\d{1,2})(\\d{2})","$2"))
    df = df.withColumn("CRSDepHour",f.regexp_replace(f.col("CRSDepTime"),"(\\d{1,2})(\\d{2})","$1"))
    df = df.withColumn("CRSDepMinute",f.regexp_replace(f.col("CRSDepTime"),"(\\d{1,2})(\\d{2})","$2"))
    df = df.withColumn("CRSArrHour",f.regexp_replace(f.col("CRSArrTime"),"(\\d{1,2})(\\d{2})","$1"))
    df = df.withColumn("CRSArrMinute",f.regexp_replace(f.col("CRSArrTime"),"(\\d{1,2})(\\d{2})","$2"))
    df = df.drop("Year") #Year is drop since each dataset only contains info for one year.
   
    return df


def clean_data2(df):
    df = df.dropna(subset=["ArrDelay","TailNum"]) #If no flight useless for ArrDelay
    df = df.drop("Cancelled","CancellationCode") #After removing every missing for ArrDelay Cancelled is always 0 and cancellation code is null. Both dropped. No additional information
    return df

'''def drop_null(df):
    df= df.na.drop()
    return df'''



#########################################   ORGANISE DATA   ##############################
def organise_data(df):
    #Function to organise data in a more understandable way
    df = df[["Date","DayOfWeek","DepTime","CRSDepTime","CRSArrTime","UniqueCarrier","TailNum","CRSElapsedTime",\
        "DepDelay","Origin","Dest","Distance","TaxiOut","ArrDelay"]] 
    return df
    




#########################################   CAST DATA   ##############################
def cast_data(df):
    #Casting of variables to propper datatypes. Originally all of them are strings. True strings like UniqueCarrier are left as they came.
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
    #Function to show missing values for categorical and string data. All distinct values are obtained and then counted.
    print("\nDifferent carriers: ",df.select("UniqueCarrier").distinct().count())
    print("\nDifferent TailNum: ",df.select("TailNum").distinct().count())
    print("\nDifferent Origin: ",df.select("Origin").distinct().count())
    print("\nDifferent Destination: ",df.select("dest").distinct().count())





#########################################   SHOW MISSING VALUES   ##############################
def show_missing_values(df):
    #A function to show missing values in each variable. Used to check if all missing values where really removed.
    df.select([f.count(f.when(f.col(c).isNull(), c)).alias(c) for c in df.columns]).show()




#########################################   COMPUTE CORRELATIONS   ##############################
def var_correlations(df):
    #Function to compute correlations among numerical attributes
    num_mask = [var_type=='double' for (var_name,var_type) in df.dtypes] #We look for columns whose values are double in the dtypes of the dataframe. A vector of boolean indicating which variable is numerical
    num_cols = [df.columns[i] for i in range(len(df.columns)) if num_mask[i]]  #We obtain column names for numerical values. (If num_mask == true for that column)
    df_num = df[num_cols]    #A new dataframe is obtained containing only numerical columns
    corr = [[df_num.corr(c1,c2) for c1 in df_num.columns] for c2 in df_num.columns] #Correlations between each pair of columns is computed
    print(num_cols)
    [print("\n",corr_row) for corr_row in corr]
    return corr




#########################################   CAT TO NUM   ##############################
def cat_to_num(df):
    columns = ["DayOfWeek","Month","UniqueCarrier","TailNum","Origin","Dest"] #A list containing only categorical variables column names
    indexers = [StringIndexer(inputCol=col,outputCol=col+"_index").setHandleInvalid("skip") for col in columns] #Create an StringIndexer for each categorical attribute.
    #Error with some TailNum since we fit on train and, if not present in that set and then appears in train our fitted pipeline does not know what to do. Solution setHandleInvalid("skip")

    ohe_input_cols = [col+"_index" for col in columns] #Create a list with the column names of indexed variables (the output columns of StringIndexer)
    ohe_output_cols = [col+"_vect" for col in columns] #Creat a list of column names to serve as the output of OneHotEncoder
    ohe = OneHotEncoder(inputCols=ohe_input_cols,outputCols=ohe_output_cols) #OneHotEncoder transformation for the whole dataset

    return indexers+[ohe] #A list of stages to define the pipeline is returned





#########################################   FEATURE SELECTION  ##############################
def feature_subset_selection(df):
    columns = ["DayOfWeek_vect","CRSElapsedTime","DepDelay","Distance","TaxiOut","DayOfMonth","DepHour","DepMinute","CRSDepHour","CRSDepMinute",
                "CRSArrHour","CRSArrMinute","Month_vect","UniqueCarrier_vect","TailNum_vect","Origin_vect","Dest_vect"] #A list with all available features is created
    assembler = VectorAssembler(inputCols=columns,outputCol="Features") #VectorAssembler to convert the set of available columns into a single vector column containing all this information
    ufss = UnivariateFeatureSelector(featuresCol="Features",outputCol="selectedFeatures",
                                    labelCol="ArrDelay",selectionMode="fpr") #UFSS with input the vector column created by VectorAssembler and output column a new vector column containing only selected features
    ufss.setFeatureType("continuous").setLabelType("continuous").setSelectionThreshold(0.05) #Threshold for selecting features is set to 0.05
    return [assembler,ufss] #A list of stages to define the pipeline is returned
    




#########################################   PIPELINE  ##############################
def create_pipeline(stages):
    pipeline=Pipeline().setStages(stages) #A pipeline is created with the list of stages passed as parameter
    return pipeline

def fit_pipeline(pipeline,df):
    pipeline = pipeline.fit(df) #The pipelined passed as parameter is fit over the dataframe also passed as a parameter  
    return pipeline

def apply_pipeline(pipeline,df):
    results = pipeline.transform(df) #The pipeline passed as parameter is used to transform the dataframe also passed as a parameter
    return results




#########################################   MODEL METRICS  ##############################
def print_metrics_LR(regressor):
    
    print("Linear Regression")

    # What is the relationship between my dependent (x) and independent (y) variable? 
    print("Coefficients: %s" % regressor.coefficients)
    
    # How much of the data does this model explain?
    print("R-squared: %f" % regressor.summary.r2)    
    
    # square root of the average of the squared difference of the predicted and actual value
    print("RMSE Root Mean Square Error: %f" % regressor.summary.rootMeanSquaredError)
    

def print_metrics_DT(model):
    # Print the maximum depth of the best Tree model
    print("Maximum depth: ",model.maxDepth)


def print_metrics_LR(model):
    
    print("Linear Regression. Best Model ")

    # What is the relationship between my dependent (x) and independent (y) variable? 
    print("Coefficients: %s" % model.coefficients)
    
    if model.hasSummary:
        # How much of the data does this model explain?
        print("R-squared: %f" % model.summary.r2)    
    
        # square root of the average of the squared difference of the predicted and actual value

        print("RMSE: %f" % model.summary.rootMeanSquaredError)   



def prints_metrics_GLR(model):
    # Print the coefficients and intercept for generalized linear regression model
    print("Coefficients: " + str(model.coefficients))
    print("Intercept: " + str(model.intercept))

    # Summarize the model over the training set and print out some metrics
    if model.hasSummary:
        summary = model.summary
        print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
        print("T Values: " + str(summary.tValues))
        print("P Values: " + str(summary.pValues))
        print("Dispersion: " + str(summary.dispersion))
        print("Null Deviance: " + str(summary.nullDeviance))
        print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
        print("Deviance: " + str(summary.deviance))
        print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
        print("AIC: " + str(summary.aic))
        print("Deviance Residuals: ")
        summary.residuals().show()