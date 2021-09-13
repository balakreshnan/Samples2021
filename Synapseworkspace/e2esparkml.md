# Azure Synapse Analytics - End to End for Automated ML using Azure Machine learning

## Move Data, Process Data, Train a model using Azure Synapse Analytics Workspace

## Unified one platform for Data and Data Science life cycle management to increase productivity

## Note

- This article is to show the functionality

## Prerequistie

- Azure Account
- Azure synapse analytics workspace
- Azure Machine learning workspace
- Azure SQL Database
- Azure Storage
- Download Covid19 data from kaggel and import into Azure SQL as external source simulated data
- Create one table as dbo.covid19data and upload the data

## Architecture

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml18.jpg "Service Health")

- The above architecture flow has 3 components
- Copy Activity to move data from external source and bring to raw or input zone
- Next is Data Flow activity to do drag and drop ETL/ELT to transform or process data and move to final
- Notebook to run automated ML sdk using spark to run Machine learning model
- We are using regression to predict the deaths occuring due to covid 19

## Steps

- Log into Azure Synapse workspace
- Create a pipeline/integrate
- Drag and Drop Copy Activity
- for Source select the above SQL server - (need to create linked services)

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml19.jpg "Service Health")

- Now go to source and create a parquet source in default synapse storage

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml3.jpg "Service Health")

- Details of underlying storage linked service

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml4.jpg "Service Health")

- Next create a data flow to process data
- For sample i am using delta to make CDC easier
- Create a source with the copy activities destination

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml5.jpg "Service Health")

- We can resue the copy activity destination linked service it self
- Now Drag the select

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml6.jpg "Service Health")

- Select all columns
- Enable debug and view the results. This dataset is open source so no PII

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml7.jpg "Service Health")

- Then drag the alter row to enable delta processing

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml8.jpg "Service Health")

- Configure the parition as single for testing

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml9.jpg "Service Health")

- Drag the sink

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml10.jpg "Service Health")

- Configure the output as inline dataset and select delta

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml11.jpg "Service Health")

- Select the columns
- See the debug to validate data is processed

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml12.jpg "Service Health")

- Configure the data flow in integrate or pipeline

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml13.jpg "Service Health")

- Now to Machine learning
- Create a notebook
- Access the above sink data set as input for modelling
- Create a notebook

## Code

- Pull the data

```
%%pyspark
df = spark.read.load('abfss://container@storageaccount.dfs.core.windows.net/covid19aggroutput/*.parquet', format='parquet')
display(df.limit(10))
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml14.jpg "Service Health")

- Print Schema

```
df.printSchema
```

- import necessary

```
from pyspark.sql.functions import *
from pyspark.sql import *
```

- convert date to date data type

```
df1 = df.withColumn("Date", to_date("ObservationDate", "MM/dd/yyyy")) 
display(df1)
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml15.jpg "Service Health")

- Create new columns

```
df2 = df1.withColumn("year", year(col("Date"))).withColumn("month", month(col("Date"))).withColumn("day", dayofmonth(col("Date")))
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml16.jpg "Service Health")

- Next is import ML libary
- get only the necessary columns

```
dffinal = df2[["year","month", "day", "Confirmed", "Deaths", "Recovered"]]
```

- Spark ML starts here

```
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ["year","month", "day", "Confirmed", "Recovered"], outputCol = 'features')
mldf = vectorAssembler.transform(dffinal)
mldf = mldf.select(['features', 'Deaths'])
mldf.show(3)
```

- Split the training and test data set

```
splits = mldf.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]
```

- Configure the model

```
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='Deaths', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))
```

- Print the model accruacy

```
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
```

- Evaluate with test data

```
lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","Deaths","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="Deaths",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
```

- display results

```
test_result = lr_model.evaluate(test_df)
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)
```

- now run predictions

```
predictions = lr_model.transform(test_df)
predictions.select("prediction","Deaths","features").show()
```

- remember the model output is not our aim of this tutorial is for. Just to show you the process.
- Save the pipeline and run the debug

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml20.jpg "Service Health")