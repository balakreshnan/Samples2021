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

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml1.jpg "Service Health")

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

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml2.jpg "Service Health")

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

```
import azureml.core

from azureml.core import Experiment, Workspace, Dataset, Datastore
from azureml.train.automl import AutoMLConfig
from azureml.data.dataset_factory import TabularDatasetFactory
```

- Configure the Azure machine learning workspace
- Set the configuration to access Automated Machine Learning

```
subscription_id = "xxxxxxxx"
resource_group = "xxxxxx"
workspace_name = "workspacename"
experiment_name = "nameofsynapse-covid19aggr-20210913"

ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
experiment = Experiment(ws, experiment_name)
```

- get only the necessary columns

```
dffinal = df2[["year","month", "day", "Confirmed", "Deaths", "Recovered"]]
```

- Register the data store

```
datastore = Datastore.get_default(ws)
dataset = TabularDatasetFactory.register_spark_dataframe(df, datastore, name = experiment_name + "-dataset")
```

- Configure automated ml

```
automl_config = AutoMLConfig(spark_context = sc,
                             task = "regression",
                             training_data = dataset,
                             label_column_name = "Deaths",
                             primary_metric = "spearman_correlation",
                             experiment_timeout_hours = 1,
                             max_concurrent_iterations = 4,
                             enable_onnx_compatible_models = False)
```

- Run Experiment

```
run = experiment.submit(automl_config)
```

- Wait for completion

```
run.wait_for_completion(show_output=False)
```

- Display the output

```
displayHTML("<a href={} target='_blank'>Your experiment in Azure Machine Learning portal: {}</a>".format(run.get_portal_url(), run.id))
```

- run the experiment and document the output

```
run.wait_for_completion()

# Install required dependency
import pip
pip.main(["install", "azure-storage-blob==12.5.0"])

import mlflow

# Get best model from automl run
best_run, non_onnx_model = run.get_output()

artifact_path = experiment_name + "_artifact"

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    # Save the model to the outputs directory for capture
    mlflow.sklearn.log_model(non_onnx_model, artifact_path)

    # Register the model to AML model registry
    mlflow.register_model("runs:/" + run.info.run_id + "/" + artifact_path, "bbaccsynapse-covid19-20201217032226-Best")
```

- Save the pipeline and run the debug

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/e2eautoml17.jpg "Service Health")