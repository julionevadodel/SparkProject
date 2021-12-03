import matplotlib.pyplot as plt
import seaborn as sns

def generate_boxplot(sqlcontext, *data: str):
    for variable in data:
        data = sqlcontext.sql("SELECT " + variable + " from dataframe").toPandas()
        data.boxplot()
        plt.show()

def generate_correlation_matrix(df):
    correlation = df.toPandas().corr()
    sns.heatmap(correlation, linewidht = 0.5)
    plt.show()

def generate_histogram(sqlcontext, *data: str):
    for variable in data:
        data = sqlcontext.sql("SELECT " + variable + " from dataframe").toPandas()
        data.hist()
        plt.show()

def generate_barplot(sqlcontext, *data: str):
    for variable in data:
        data = sqlcontext.sql("SELECT " + variable + " from dataframe").toPandas()
        data.barplot()
        plt.show()