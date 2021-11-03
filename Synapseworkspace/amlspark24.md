# Azure Synapse Spark using AML SDK

## Azure Synapse Spark using Azure Machine learning python SDK to use workspace resource

- Integrate Azure Machine Learning into your Spark application
- Use sdk and connect to AML workspace
- Ability to extend compute using AML Workspace compute options
- Extend Automated ML in synapse spark

## Prerequistie

- Azure Account
- Azure Storage
- Azure Machine learning Service
- Github account and repository
- Azure Service Principal Account
- Provide service principal contributor access to Machine learning resource
- Azure Keyvault to store secrets
- Update the keyvault with Service principal Secrets
- This automates the training code and Registers the model

## Steps

- Log into Azure Synpase workspace UI
- Create a new spark cluster definition with spark 2.4

```
Note: AutoML SDK only works with python 3.6 at the time this article is written Nov 3rd 2021.
In Future versions of the SDK, we will be able to use python 3.7 and above.
```

- Create a spark notebook
- Select pyspark
- Select the Spark 2.4 spark cluster you created above
- i choose 9 nodes to process with large instances

## Code

```
import azureml.core
print(azureml.core.VERSION)
```

- output was 1.32.0

## Data processing

- We are copying NYC taxi data from public storage into local
- Import AutoML libraries, this validates the version of the SDK

```
from azureml.train.automl import AutoMLConfig
```

- Now bring the service principal credentials into the workspace
- We are going to use service principal to access AML workspace
- All secrets are stored in keyvault
- Create a linked services for keyvault so that workspace can access

```
import sys
from pyspark.sql import SparkSession

sc = SparkSession.builder.getOrCreate()
token_library = sc._jvm.com.microsoft.azure.synapse.tokenlibrary.TokenLibrary

tenantid = token_library.getSecret("accvault1", "tenantid", "accvault1")
svcpid = token_library.getSecret("accvault1", "svcpid", "accvault1")
scvpsecret = token_library.getSecret("accvault1", "scvpsecret", "accvault1")
print(tenantid)
```

- Set the authentication token

```
from azureml.core.authentication import ServicePrincipalAuthentication

sp = ServicePrincipalAuthentication(tenant_id=tenantid, # tenantID
                                    service_principal_id=svcpid, # clientId
                                    service_principal_password=scvpsecret) # clientSecret
```

- Now Authenticate and access workspace

```
from azureml.core import Workspace

ws = Workspace.get(name="amlworkspacename",
                   auth=sp,
                   subscription_id="xxxxxxxxxxxxxxxxxxxxxxxx",
                   resource_group='rgname')
```

- Now bring the data into the workspace

```
blob_account_name = "azureopendatastorage"
blob_container_name = "nyctlc"
blob_relative_path = "yellow"
blob_sas_token = r""

# Allow Spark to read from the blob remotely
wasbs_path = 'wasbs://%s@%s.blob.core.windows.net/%s' % (blob_container_name, blob_account_name, blob_relative_path)
spark.conf.set('fs.azure.sas.%s.%s.blob.core.windows.net' % (blob_container_name, blob_account_name),blob_sas_token)

# Spark read parquet; note that it won't load any data yet
df = spark.read.parquet(wasbs_path)
```

- Filter data

```
# Create an ingestion filter
start_date = '2015-01-01 00:00:00'
end_date = '2015-12-31 00:00:00'

filtered_df = df.filter('tpepPickupDateTime > "' + start_date + '" and tpepPickupDateTime< "' + end_date + '"')

display(filtered_df.describe())
```

- do a total count 

```
filtered_df.count()
```

- Down sample the data

```
from datetime import datetime
from pyspark.sql.functions import *

# To make development easier, faster, and less expensive, downsample for now
sampled_taxi_df = filtered_df.sample(True, 0.001, seed=1234)
```

- Filter data if needed

```
taxi_df = sampled_taxi_df.select('vendorID', 'passengerCount', 'tripDistance',  'startLon', 'startLat', 'endLon' \
                                , 'endLat', 'paymentType', 'fareAmount', 'tipAmount'\
                                , column('puMonth').alias('month_num') \
                                , date_format('tpepPickupDateTime', 'hh').alias('hour_of_day')\
                                , date_format('tpepPickupDateTime', 'EEEE').alias('day_of_week')\
                                , dayofmonth(col('tpepPickupDateTime')).alias('day_of_month')
                                ,(unix_timestamp(col('tpepDropoffDateTime')) - unix_timestamp(col('tpepPickupDateTime'))).alias('trip_time'))\
                        .filter((sampled_taxi_df.passengerCount > 0) & (sampled_taxi_df.passengerCount < 8)\
                                & (sampled_taxi_df.tipAmount >= 0)\
                                & (sampled_taxi_df.fareAmount >= 1) & (sampled_taxi_df.fareAmount <= 250)\
                                & (sampled_taxi_df.tipAmount < sampled_taxi_df.fareAmount)\
                                & (sampled_taxi_df.tripDistance > 0) & (sampled_taxi_df.tripDistance <= 200)\
                                & (sampled_taxi_df.rateCodeId <= 5)\
                                & (sampled_taxi_df.paymentType.isin({"1", "2"})))
taxi_df.show(10)
```

- I am only choosing few rows for this testing

```
taxi_df = sampled_taxi_df.limit(10000)
```

- Now create batch compute in AML Workspace

```
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

#ws = Workspace.from_config() # This automatically looks for a directory .azureml

# Choose a name for your CPU cluster
cpu_cluster_name = "cpu-cluster"

# Verify that the cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                           idle_seconds_before_scaledown=2400,
                                                           min_nodes=0,
                                                           max_nodes=4)
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

cpu_cluster.wait_for_completion(show_output=True)
```

## Training

- Let's split the data for training and testing

```
# Random split dataset using Spark; convert Spark to pandas
training_data, validation_data = taxi_df.randomSplit([0.8,0.2], 223)
```

- Register the data in AML workspace
- This allows the batch compute to know where the data is

```
import pandas 
from azureml.core import Dataset

# Get the Azure Machine Learning default datastore
datastore = ws.get_default_datastore()
training_pd = training_data.toPandas().to_csv('training_pd.csv', index=False)

# Convert into an Azure Machine Learning tabular dataset
datastore.upload_files(files = ['training_pd.csv'],
                       target_path = 'train-dataset/tabular/',
                       overwrite = True,
                       show_progress = True)
dataset_training = Dataset.Tabular.from_delimited_files(path = [(datastore, 'train-dataset/tabular/training_pd.csv')])
```

- Now setup configuration for automated machine learning

```
import logging

automl_settings = {
    "iteration_timeout_minutes": 10,
    "experiment_timeout_minutes": 30,
    "enable_early_stopping": True,
    "primary_metric": 'r2_score',
    "featurization": 'auto',
    "verbosity": logging.INFO,
    "max_concurrent_iterations": 4,
    "n_cross_validations": 2}
```

- Now set up the autoML

```
automl_config = AutoMLConfig(task='regression',
                             debug_log='automated_ml_errors.log',
                             training_data = dataset_training,
                             #spark_context = sc,
                             model_explainability = True, 
                             compute_target=cpu_cluster,
                             #environment=myenv,
                             label_column_name ="fareAmount",**automl_settings)
```

- you can see the compute_target
- This is what determines what compute to use
- in the above case will be AML batch compute or called compute cluster in AML workspace

- Run the model

```
from azureml.core.experiment import Experiment

# Start an experiment in Azure Machine Learning
experiment = Experiment(ws, "aml-synapse-regression")
tags = {"Synapse": "regression"}
remote_run = experiment.submit(automl_config, show_output=True, tags = tags)

# Use the get_details function to retrieve the detailed output for the run.
run_details = remote_run.get_details()
```

## Test and Validation

- Now get the best result

```
# Get best model
best_run, fitted_model = remote_run.get_output()
```

- this section is for folks who want to get existing run

```
from azureml.core import Experiment, Workspace

Experiment = ws.experiments["aml-synapse-regression"]
```

```
from azureml.train.automl.run import AutoMLRun
#ws = Workspace.from_config()
experiment = ws.experiments['aml-synapse-regression']
automl_run = AutoMLRun(experiment, run_id = 'AutoML_xxxxxx')
```

```
best_run, fitted_model = automl_run.get_output()
```

```
validation_data_pd = validation_data.toPandas()
```

- if there is error in valdiation for the data set then convert the below column

```
validation_data_pd['storeAndFwdFlag'] = validation_data_pd['storeAndFwdFlag'].astype(bool)
```

- Run prediction with validation data set or test data set

```
#validation_data_pd = validation_data.toPandas()
y_test = validation_data_pd.pop("fareAmount").to_frame()
y_predict = fitted_model.predict(validation_data_pd)
```

- caluclate the error and accuracy score and print

```
from sklearn.metrics import mean_squared_error
from math import sqrt

# Calculate root-mean-square error
y_actual = y_test.values.flatten().tolist()
rmse = sqrt(mean_squared_error(y_actual, y_predict))

print("Root Mean Square Error:")
print(rmse)
```

- Calculate mean-absolute-percent error and model accuracy

```
# Calculate mean-absolute-percent error and model accuracy 
sum_actuals = sum_errors = 0

for actual_val, predict_val in zip(y_actual, y_predict):
    abs_error = actual_val - predict_val
    if abs_error < 0:
        abs_error = abs_error * -1

    sum_errors = sum_errors + abs_error
    sum_actuals = sum_actuals + actual_val

mean_abs_percent_error = sum_errors / sum_actuals

print("Model MAPE:")
print(mean_abs_percent_error)
print()
print("Model Accuracy:")
print(1 - mean_abs_percent_error)
```

- plot the output

```
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Calculate the R2 score by using the predicted and actual fare prices
y_test_actual = y_test["fareAmount"]
r2 = r2_score(y_test_actual, y_predict)

# Plot the actual versus predicted fare amount values
plt.style.use('ggplot')
plt.figure(figsize=(10, 7))
plt.scatter(y_test_actual,y_predict)
plt.plot([np.min(y_test_actual), np.max(y_test_actual)], [np.min(y_test_actual), np.max(y_test_actual)], color='lightblue')
plt.xlabel("Actual Fare Amount")
plt.ylabel("Predicted Fare Amount")
plt.title("Actual vs Predicted Fare Amount R^2={}".format(r2))
plt.show()
```

- Finally register the model for inferencing use

```
description = 'My automated ML model'
model_path='outputs/model.pkl'
model = best_run.register_model(model_name = 'NYCGreenTaxiModel', model_path = model_path, description = description)
print(model.name, model.version)
```