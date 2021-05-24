# Azure Synapse Workspace End to End Machine Learning

## This is a end to end sample demo to show case how spark, Dedicated sql pools and machine learning

## Use Case

- Data engineering and ETL using SQL Dedicated Pools
- Data engineering and ETL using Synapse Spark
- Show case how to invoke pipeline
- Show to orchestrate end to end

## Steps

- End to end architecture
- Show case how we can mismatch spark, SQL, Machine learning in scale

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/etl1.jpg "Service Health")

- End to End explained
- Resume Pipeline to start the dedicated pool
- Use HTTP Activity
- Use method: POST

```
https://management.azure.com/subscriptions/subscriptionid/resourceGroups/resourcegroup/providers/Microsoft.Synapse/workspaces/workspacename/sqlPools/dedicatedpoolname/resume?api-version=2019-06-01-preview
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/etl2.jpg "Service Health")

- Now clear the aggregate table to load the new one
- I am using stored procedure to clean the data

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/etl4.jpg "Service Health")

```
DROP PROCEDURE [dbo].[dropdailyaggr]
GO

SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE PROC [dbo].[dropdailyaggr] AS
Drop Table [wwi].[dailyaggr]
GO
```

- nycyellow ETL/Data engineering pipeline sample using pyspark
- Using pyspark

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/etl5.jpg "Service Health")

```
from azureml.opendatasets import NycTlcYellow

data = NycTlcYellow()
data_df = data.to_spark_dataframe()
# Display 10 rows
display(data_df.limit(10))
```

```
from pyspark.sql.functions import *
from pyspark.sql import *
```

```
df1 = data_df.withColumn("Date", (col("tpepPickupDateTime").cast("date"))) 
display(df1)
```

```
df2 = df1.withColumn("year", year(col("date"))) .withColumn("month", month(col("date"))) .withColumn("day", dayofmonth(col("date"))) .withColumn("hour", hour(col("date")))
```

```
dfgrouped = df2.groupBy("year","month").agg(sum("fareAmount").alias("Total"),count("vendorID").alias("Count")).sort(asc("year"), asc("month"))
```

```
dfgrouped.repartition(1).write.mode('overwrite').parquet("/dailyaggr/parquet/dailyaggr.parquet")
```

```
dfgrouped.repartition(1).write.mode('overwrite').option("header","true").csv("/dailyaggrcsv/csv/dailyaggr.csv")
```

```
df2.createOrReplaceTempView("nycyellow")
```

```
%%sql
select * from nycyellow limit 100
```

```
%%sql
select  year(cast(tpepPickupDateTime  as timestamp)) as tsYear,
        month(cast(tpepPickupDateTime  as timestamp)) as tsmonth,
        day(cast(tpepPickupDateTime  as timestamp)) as tsDay, 
        hour(cast(tpepPickupDateTime  as timestamp)) as tsHour,
        avg(totalAmount) as avgTotal, avg(fareAmount) as avgFare
from nycyellow
group by  tsYear, tsmonth,tsDay, tsHour
order by  tsYear, tsmonth,tsDay, tsHour
```

```
%%sql
DROP TABLE dailyaggr
```

```
%%sql
CREATE TABLE dailyaggr
  COMMENT 'This table is created with existing data'
  AS select  year(cast(tpepPickupDateTime  as timestamp)) as tsYear,
        month(cast(tpepPickupDateTime  as timestamp)) as tsmonth,
        day(cast(tpepPickupDateTime  as timestamp)) as tsDay, 
        hour(cast(tpepPickupDateTime  as timestamp)) as tsHour,
        avg(totalAmount) as avgTotal, avg(fareAmount) as avgFare
from nycyellow
group by  tsYear, tsmonth,tsDay, tsHour
order by  tsYear, tsmonth,tsDay, tsHour
```

```
dailyaggr = spark.sql("SELECT tsYear, tsMonth, tsDay, tsHour, avgTotal FROM dailyaggr")
display(dailyaggr)
```

```
%%spark
import com.microsoft.spark.sqlanalytics.utils.Constants
import org.apache.spark.sql.SqlAnalyticsConnector._
```

```
from pyspark.ml.regression import LinearRegression
```

```
%%pyspark 
import pyspark 
print(print(pyspark.__version__))
```

```
%%spark
import org.apache.spark.ml.feature.VectorAssembler 
import org.apache.spark.ml.linalg.Vectors 
val dailyaggr = spark.sql("SELECT tsYear, tsMonth, tsDay, tsHour, avgTotal FROM dailyaggr")
val featureCols=Array("tsYear","tsMonth","tsDay","tsHour") 
val assembler: org.apache.spark.ml.feature.VectorAssembler= new VectorAssembler().setInputCols(featureCols).setOutputCol("features") 
val assembledDF = assembler.setHandleInvalid("skip").transform(dailyaggr) 
val assembledFinalDF = assembledDF.select("avgTotal","features")
```

```
%%spark
import com.microsoft.spark.sqlanalytics.utils.Constants
import org.apache.spark.sql.SqlAnalyticsConnector._
```

```
%%spark
dailyaggr.repartition(2).write.synapsesql("accsynapsepools.wwi.dailyaggr", Constants.INTERNAL)
```

```
%%spark
import org.apache.spark.ml.feature.Normalizer 
val normalizedDF = new Normalizer().setInputCol("features").setOutputCol("normalizedFeatures").transform(assembledFinalDF)
```

```
%%spark
val normalizedDF1 = normalizedDF.na.drop()
```

```
%%spark
val Array(trainingDS, testDS) = normalizedDF1.randomSplit(Array(0.7, 0.3))
```

```
%%spark
import org.apache.spark.ml.regression.LinearRegression
// Create a LinearRegression instance. This instance is an Estimator. 
val lr = new LinearRegression().setLabelCol("avgTotal").setMaxIter(100)
// Print out the parameters, documentation, and any default values. println(s"Linear Regression parameters:\n ${lr.explainParams()}\n") 
// Learn a Linear Regression model. This uses the parameters stored in lr.
val lrModel = lr.fit(trainingDS)
// Make predictions on test data using the Transformer.transform() method.
// LinearRegression.transform will only use the 'features' column. 
val lrPredictions = lrModel.transform(testDS)
```

```
%%spark
import org.apache.spark.sql.functions._ 
import org.apache.spark.sql.types._ 
println("\nPredictions : " ) 
lrPredictions.select($"avgTotal".cast(IntegerType),$"prediction".cast(IntegerType)).orderBy(abs($"prediction"-$"avgTotal")).distinct.show(15)
```

```
%%spark
import org.apache.spark.ml.evaluation.RegressionEvaluator 

val evaluator_r2 = new RegressionEvaluator().setPredictionCol("prediction").setLabelCol("avgTotal").setMetricName("r2") 
//As the name implies, isLargerBetter returns if a larger value is better or smaller for evaluation. 
val isLargerBetter : Boolean = evaluator_r2.isLargerBetter 
println("Coefficient of determination = " + evaluator_r2.evaluate(lrPredictions))
```

```
%%spark
//Evaluate the results. Calculate Root Mean Square Error 
val evaluator_rmse = new RegressionEvaluator().setPredictionCol("prediction").setLabelCol("avgTotal").setMetricName("rmse") 
//As the name implies, isLargerBetter returns if a larger value is better for evaluation. 
val isLargerBetter1 : Boolean = evaluator_rmse.isLargerBetter 
println("Root Mean Square Error = " + evaluator_rmse.evaluate(lrPredictions))
```

```
%%spark
val dailyaggrdf = spark.read.synapsesql("accsynapsepools.wwi.dailyaggr")
```

```
%%spark
display(dailyaggrdf)
```

```
%%spark
dailyaggrdf.count()
```

- nyc holidays in scala code

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/etl6.jpg "Service Health")

```
// Load nyc green taxi trip records from azure open dataset
val blob_account_name = "azureopendatastorage"

val nyc_blob_container_name = "nyctlc"
val nyc_blob_relative_path = "green"
val nyc_blob_sas_token = ""

val nyc_wasbs_path = f"wasbs://$nyc_blob_container_name@$blob_account_name.blob.core.windows.net/$nyc_blob_relative_path"
spark.conf.set(f"fs.azure.sas.$nyc_blob_container_name.$blob_account_name.blob.core.windows.net",nyc_blob_sas_token)

val nyc_tlc = spark.read.parquet(nyc_wasbs_path)
```

```
// Filter data by time range
import java.sql.Timestamp
import org.joda.time.DateTime

val end_date = new Timestamp(DateTime.parse("2018-06-06").getMillis)
val start_date = new Timestamp(DateTime.parse("2018-05-01").getMillis)

val nyc_tlc_df = nyc_tlc.filter((nyc_tlc("lpepPickupDatetime") >= start_date) && (nyc_tlc("lpepPickupDatetime") <= end_date)) 
nyc_tlc_df.show(5, truncate = false)
```

```
// Extract month, day of month, and day of week from pickup datetime and add a static column for the country code to join holiday data. 
import org.apache.spark.sql.functions._

val nyc_tlc_df_expand = (
                        nyc_tlc_df.withColumn("datetime", to_date(col("lpepPickupDatetime")))
                                  .withColumn("month_num",month(col("lpepPickupDatetime")))
                                  .withColumn("day_of_month",dayofmonth(col("lpepPickupDatetime")))
                                  .withColumn("day_of_week",dayofweek(col("lpepPickupDatetime")))
                                  .withColumn("hour_of_day",hour(col("lpepPickupDatetime")))
                                  .withColumn("country_code",lit("US"))
                        )
```

```
// Remove unused columns from nyc green taxi data
val nyc_tlc_df_clean = nyc_tlc_df_expand.drop(
                    "lpepDropoffDatetime", "puLocationId", "doLocationId", "pickupLongitude", 
                     "pickupLatitude", "dropoffLongitude","dropoffLatitude" ,"rateCodeID", 
                     "storeAndFwdFlag","paymentType", "fareAmount", "extra", "mtaTax",
                     "improvementSurcharge", "tollsAmount", "ehailFee", "tripType" )
```

```
// Display 5 rows
nyc_tlc_df_clean.show(5, truncate = false)
```

```
// Load public holidays data from azure open dataset
val hol_blob_container_name = "holidaydatacontainer"
val hol_blob_relative_path = "Processed"
val hol_blob_sas_token = ""

val hol_wasbs_path = f"wasbs://$hol_blob_container_name@$blob_account_name.blob.core.windows.net/$hol_blob_relative_path"
spark.conf.set(f"fs.azure.sas.$hol_blob_container_name.$blob_account_name.blob.core.windows.net",hol_blob_sas_token)

val hol_raw = spark.read.parquet(hol_wasbs_path)

// Filter data by time range
val hol_df = hol_raw.filter((hol_raw("date") >= start_date) && (hol_raw("date") <= end_date))

// Display 5 rows
// hol_df.show(5, truncate = false)
```

```
val hol_df_clean = (
                hol_df.withColumnRenamed("countryRegionCode","country_code")
                .withColumn("datetime",to_date(col("date")))
                )

hol_df_clean.show(5, truncate = false)
```

```
// enrich taxi data with holiday data
val nyc_taxi_holiday_df = nyc_tlc_df_clean.join(hol_df_clean, Seq("datetime", "country_code") , "left")

nyc_taxi_holiday_df.show(5,truncate = false)
```

```
// Create a temp table and filter out non empty holiday rows

nyc_taxi_holiday_df.createOrReplaceTempView("nyc_taxi_holiday_df")
val result = spark.sql("SELECT * from nyc_taxi_holiday_df WHERE holidayName is NOT NULL ")
result.show(5, truncate = false)
```

```
// Load weather data from azure open dataset
val weather_blob_container_name = "isdweatherdatacontainer"
val weather_blob_relative_path = "ISDWeather/"
val weather_blob_sas_token = ""

val weather_wasbs_path = f"wasbs://$weather_blob_container_name@$blob_account_name.blob.core.windows.net/$weather_blob_relative_path"
spark.conf.set(f"fs.azure.sas.$weather_blob_container_name.$blob_account_name.blob.core.windows.net",hol_blob_sas_token)

val isd = spark.read.parquet(weather_wasbs_path)

// Display 5 rows
// isd.show(5, truncate = false)
```

```
// Filter data by time range
val isd_df = isd.filter((isd("datetime") >= start_date) && (isd("datetime") <= end_date))

// Display 5 rows
isd_df.show(5, truncate = false)
```

```
al weather_df = (
                isd_df.filter(isd_df("latitude") >= "40.53")
                        .filter(isd_df("latitude") <= "40.88")
                        .filter(isd_df("longitude") >= "-74.09")
                        .filter(isd_df("longitude") <= "-73.72")
                        .filter(isd_df("temperature").isNotNull)
                        .withColumnRenamed("datetime","datetime_full")
                        )
```

```
// Remove unused columns
val weather_df_clean = weather_df.drop("usaf", "wban", "longitude", "latitude").withColumn("datetime", to_date(col("datetime_full")))

//weather_df_clean.show(5, truncate = false)
```

```
val weather_df_grouped = (
                        weather_df_clean.groupBy('datetime).
                        agg(
                            mean('snowDepth) as "avg_snowDepth",
                            max('precipTime) as "max_precipTime",
                            mean('temperature) as "avg_temperature",
                            max('precipDepth) as "max_precipDepth"
                            )
                        )

weather_df_grouped.show(5, truncate = false)
```

```
// Enrich taxi data with weather
val nyc_taxi_holiday_weather_df = nyc_taxi_holiday_df.join(weather_df_grouped, Seq("datetime") ,"left")
nyc_taxi_holiday_weather_df.cache()
```

```
nyc_taxi_holiday_weather_df.show(5,truncate = false)
```

```
// Run the describe() function on the new dataframe to see summary statistics for each field.
display(nyc_taxi_holiday_weather_df.describe())
```

```
nyc_taxi_holiday_weather_df.count
```

```
// Remove invalid rows with less than 0 taxi fare or tip
val final_df = (
            nyc_taxi_holiday_weather_df.
            filter(nyc_taxi_holiday_weather_df("tipAmount") > 0).
            filter(nyc_taxi_holiday_weather_df("totalAmount") > 0)
            )
```

```
spark.sql("DROP TABLE IF EXISTS NYCTaxi.nyc_taxi_holiday_weather"); 
```

```
spark.sql("DROP DATABASE IF EXISTS NYCTaxi"); 
spark.sql("CREATE DATABASE NYCTaxi"); 
spark.sql("USE NYCTaxi");
```

```
final_df.write.saveAsTable("nyc_taxi_holiday_weather");
val final_results = spark.sql("SELECT COUNT(*) FROM nyc_taxi_holiday_weather");
final_results.show(5, truncate = false)
```

- Pause pipeline to pause dedicated pool
- Use HTTP Activity
- Use method: POST

```
https://management.azure.com/subscriptions/subscriptionid/resourceGroups/resourcegroup/providers/Microsoft.Synapse/workspaces/workdpspacename/sqlPools/dedicatedpoolname/pause?api-version=2019-06-01-preview
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/etl3.jpg "Service Health")

- Azure ML Spark

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/etl7.jpg "Service Health")

```
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil import parser
from pyspark.sql.functions import unix_timestamp, date_format, col, when
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.feature import RFormula
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
```

```
from azureml.opendatasets import NycTlcYellow

end_date = parser.parse('2018-06-06')
start_date = parser.parse('2018-05-01')
nyc_tlc = NycTlcYellow(start_date=start_date, end_date=end_date)
filtered_df = nyc_tlc.to_spark_dataframe()
```

```
# To make development easier, faster and less expensive down sample for now
sampled_taxi_df = filtered_df.sample(True, 0.001, seed=1234)
```

```
#sampled_taxi_df.show(5)
display(sampled_taxi_df)
```

```
sampled_taxi_df.createOrReplaceTempView("nytaxi")
```

```
taxi_df = sampled_taxi_df.select('totalAmount', 'fareAmount', 'tipAmount', 'paymentType', 'rateCodeId', 'passengerCount'\
                                , 'tripDistance', 'tpepPickupDateTime', 'tpepDropoffDateTime'\
                                , date_format('tpepPickupDateTime', 'hh').alias('pickupHour')\
                                , date_format('tpepPickupDateTime', 'EEEE').alias('weekdayString')\
                                , (unix_timestamp(col('tpepDropoffDateTime')) - unix_timestamp(col('tpepPickupDateTime'))).alias('tripTimeSecs')\
                                , (when(col('tipAmount') > 0, 1).otherwise(0)).alias('tipped')
                                )\
                        .filter((sampled_taxi_df.passengerCount > 0) & (sampled_taxi_df.passengerCount < 8)\
                                & (sampled_taxi_df.tipAmount >= 0) & (sampled_taxi_df.tipAmount <= 25)\
                                & (sampled_taxi_df.fareAmount >= 1) & (sampled_taxi_df.fareAmount <= 250)\
                                & (sampled_taxi_df.tipAmount < sampled_taxi_df.fareAmount)\
                                & (sampled_taxi_df.tripDistance > 0) & (sampled_taxi_df.tripDistance <= 100)\
                                & (sampled_taxi_df.rateCodeId <= 5)
                                & (sampled_taxi_df.paymentType.isin({"1", "2"}))
                                )
```

```
taxi_featurised_df = taxi_df.select('totalAmount', 'fareAmount', 'tipAmount', 'paymentType', 'passengerCount'\
                                                , 'tripDistance', 'weekdayString', 'pickupHour','tripTimeSecs','tipped'\
                                                , when((taxi_df.pickupHour <= 6) | (taxi_df.pickupHour >= 20),"Night")\
                                                .when((taxi_df.pickupHour >= 7) & (taxi_df.pickupHour <= 10), "AMRush")\
                                                .when((taxi_df.pickupHour >= 11) & (taxi_df.pickupHour <= 15), "Afternoon")\
                                                .when((taxi_df.pickupHour >= 16) & (taxi_df.pickupHour <= 19), "PMRush")\
                                                .otherwise(0).alias('trafficTimeBins')
                                              )\
                                       .filter((taxi_df.tripTimeSecs >= 30) & (taxi_df.tripTimeSecs <= 7200))
```

```
# Since the sample uses an algorithm that only works with numeric features, convert them so they can be consumed
sI1 = StringIndexer(inputCol="trafficTimeBins", outputCol="trafficTimeBinsIndex")
en1 = OneHotEncoder(dropLast=False, inputCol="trafficTimeBinsIndex", outputCol="trafficTimeBinsVec")
sI2 = StringIndexer(inputCol="weekdayString", outputCol="weekdayIndex")
en2 = OneHotEncoder(dropLast=False, inputCol="weekdayIndex", outputCol="weekdayVec")

# Create a new dataframe that has had the encodings applied
encoded_final_df = Pipeline(stages=[sI1, en1, sI2, en2]).fit(taxi_featurised_df).transform(taxi_featurised_df)
```

```
#Decide on the split between training and testing data from the dataframe
trainingFraction = 0.7
testingFraction = (1-trainingFraction)
seed = 1234

# Split the dataframe into test and training dataframes
train_data_df, test_data_df = encoded_final_df.randomSplit([trainingFraction, testingFraction], seed=seed)
```

```
## Create a new LR object for the model
logReg = LogisticRegression(maxIter=10, regParam=0.3, labelCol = 'tipped')

## The formula for the model
classFormula = RFormula(formula="tipped ~ pickupHour + weekdayVec + passengerCount + tripTimeSecs + tripDistance + fareAmount + paymentType+ trafficTimeBinsVec")

## Undertake training and create an LR model
lrModel = Pipeline(stages=[classFormula, logReg]).fit(train_data_df)

## Saving the model is optional but its another form of inter session cache
datestamp = datetime.now().strftime('%m-%d-%Y-%s')
fileName = "lrModel_" + datestamp
logRegDirfilename = fileName
lrModel.save(logRegDirfilename)

## Predict tip 1/0 (yes/no) on the test dataset, evaluation using AUROC
predictions = lrModel.transform(test_data_df)
predictionAndLabels = predictions.select("label","prediction").rdd
metrics = BinaryClassificationMetrics(predictionAndLabels)
print("Area under ROC = %s" % metrics.areaUnderROC)
```

```
## Plot the ROC curve, no need for pandas as this uses the modelSummary object
modelSummary = lrModel.stages[-1].summary

plt.plot([0, 1], [0, 1], 'r--')
plt.plot(modelSummary.roc.select('FPR').collect(),
         modelSummary.roc.select('TPR').collect())
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
```

- Spark MLLib modelling

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/etl8.jpg "Service Health")

```
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil import parser
from pyspark.sql.functions import unix_timestamp, date_format, col, when
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.feature import RFormula
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
```

```
from azureml.opendatasets import NycTlcYellow

end_date = parser.parse('2018-06-06')
start_date = parser.parse('2018-05-01')
nyc_tlc = NycTlcYellow(start_date=start_date, end_date=end_date)
filtered_df = nyc_tlc.to_spark_dataframe()
```

```
# To make development easier, faster and less expensive down sample for now
sampled_taxi_df = filtered_df.sample(True, 0.001, seed=1234)
```

```
#sampled_taxi_df.show(5)
display(sampled_taxi_df)
```

```
sampled_taxi_df.createOrReplaceTempView("nytaxi")
```

```
taxi_df = sampled_taxi_df.select('totalAmount', 'fareAmount', 'tipAmount', 'paymentType', 'rateCodeId', 'passengerCount'\
                                , 'tripDistance', 'tpepPickupDateTime', 'tpepDropoffDateTime'\
                                , date_format('tpepPickupDateTime', 'hh').alias('pickupHour')\
                                , date_format('tpepPickupDateTime', 'EEEE').alias('weekdayString')\
                                , (unix_timestamp(col('tpepDropoffDateTime')) - unix_timestamp(col('tpepPickupDateTime'))).alias('tripTimeSecs')\
                                , (when(col('tipAmount') > 0, 1).otherwise(0)).alias('tipped')
                                )\
                        .filter((sampled_taxi_df.passengerCount > 0) & (sampled_taxi_df.passengerCount < 8)\
                                & (sampled_taxi_df.tipAmount >= 0) & (sampled_taxi_df.tipAmount <= 25)\
                                & (sampled_taxi_df.fareAmount >= 1) & (sampled_taxi_df.fareAmount <= 250)\
                                & (sampled_taxi_df.tipAmount < sampled_taxi_df.fareAmount)\
                                & (sampled_taxi_df.tripDistance > 0) & (sampled_taxi_df.tripDistance <= 100)\
                                & (sampled_taxi_df.rateCodeId <= 5)
                                & (sampled_taxi_df.paymentType.isin({"1", "2"}))
                                )
```

```
taxi_featurised_df = taxi_df.select('totalAmount', 'fareAmount', 'tipAmount', 'paymentType', 'passengerCount'\
                                                , 'tripDistance', 'weekdayString', 'pickupHour','tripTimeSecs','tipped'\
                                                , when((taxi_df.pickupHour <= 6) | (taxi_df.pickupHour >= 20),"Night")\
                                                .when((taxi_df.pickupHour >= 7) & (taxi_df.pickupHour <= 10), "AMRush")\
                                                .when((taxi_df.pickupHour >= 11) & (taxi_df.pickupHour <= 15), "Afternoon")\
                                                .when((taxi_df.pickupHour >= 16) & (taxi_df.pickupHour <= 19), "PMRush")\
                                                .otherwise(0).alias('trafficTimeBins')
                                              )\
                                       .filter((taxi_df.tripTimeSecs >= 30) & (taxi_df.tripTimeSecs <= 7200))
```

```
# Since the sample uses an algorithm that only works with numeric features, convert them so they can be consumed
sI1 = StringIndexer(inputCol="trafficTimeBins", outputCol="trafficTimeBinsIndex")
en1 = OneHotEncoder(dropLast=False, inputCol="trafficTimeBinsIndex", outputCol="trafficTimeBinsVec")
sI2 = StringIndexer(inputCol="weekdayString", outputCol="weekdayIndex")
en2 = OneHotEncoder(dropLast=False, inputCol="weekdayIndex", outputCol="weekdayVec")

# Create a new dataframe that has had the encodings applied
encoded_final_df = Pipeline(stages=[sI1, en1, sI2, en2]).fit(taxi_featurised_df).transform(taxi_featurised_df)
```

```
#Decide on the split between training and testing data from the dataframe
trainingFraction = 0.7
testingFraction = (1-trainingFraction)
seed = 1234

# Split the dataframe into test and training dataframes
train_data_df, test_data_df = encoded_final_df.randomSplit([trainingFraction, testingFraction], seed=seed)
```

```
## Create a new LR object for the model
logReg = LogisticRegression(maxIter=10, regParam=0.3, labelCol = 'tipped')

## The formula for the model
classFormula = RFormula(formula="tipped ~ pickupHour + weekdayVec + passengerCount + tripTimeSecs + tripDistance + fareAmount + paymentType+ trafficTimeBinsVec")

## Undertake training and create an LR model
lrModel = Pipeline(stages=[classFormula, logReg]).fit(train_data_df)

## Saving the model is optional but its another form of inter session cache
datestamp = datetime.now().strftime('%m-%d-%Y-%s')
fileName = "lrModel_" + datestamp
logRegDirfilename = fileName
lrModel.save(logRegDirfilename)

## Predict tip 1/0 (yes/no) on the test dataset, evaluation using AUROC
predictions = lrModel.transform(test_data_df)
predictionAndLabels = predictions.select("label","prediction").rdd
metrics = BinaryClassificationMetrics(predictionAndLabels)
print("Area under ROC = %s" % metrics.areaUnderROC)
```

```
## Plot the ROC curve, no need for pandas as this uses the modelSummary object
modelSummary = lrModel.stages[-1].summary

plt.plot([0, 1], [0, 1], 'r--')
plt.plot(modelSummary.roc.select('FPR').collect(),
         modelSummary.roc.select('TPR').collect())
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
```