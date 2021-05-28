# Azure Databricks MLFlow CI/CD with Azure DevOps

## CI/CD for Machine learning model training with mlflow and batch inferencing

## Prerequistie

- Azure Account
- Azure storage account
- Azure Devops
- Azure Databricks
- Azure devOps - Databricks market place install
- Github to store code

## Data 

- Navigate to https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/ and download both winequality-red.csv and winequality-white.csv to your local machine.
- Download to local compute and upload to Azure databricks in a folder

## Steps

- First create a notebook with python
- Create a cluster with 8.2ML as runtime
- Here is the training
- Registering the model
- Use the model to batch score
- Store it back into storage as delta

```
import cloudpickle
```

- load the data

```
import pandas as pd

# In the following lines, replace <username> with your username.
white_wine = pd.read_csv("/dbfs/FileStore/shared_uploads/username/winequality_white.csv", sep=';')
red_wine = pd.read_csv("/dbfs/FileStore/shared_uploads/username/winequality_red.csv", sep=';')
```

- format the data

```
red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

# Remove spaces from column names
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
```

```
data.head()
```

- Display the chart

```
import seaborn as sns
sns.distplot(data.quality, kde=False)
```

- set data quality

```
high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality
```

- display the coreleations

```
import matplotlib.pyplot as plt

dims = (3, 4)

f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in data.columns:
  if col == 'is_red' or col == 'quality':
    continue # Box plots cannot be used on indicator variables
  sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
  axis_j += 1
  if axis_j == dims[1]:
    axis_i += 1
    axis_j = 0
```

- Drop nulls

```
data.isna().any()
```

- Split train and test

```
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, random_state=123)
X_train = train.drop(["quality"], axis=1)
X_test = test.drop(["quality"], axis=1)
y_train = train.quality
y_test = test.quality
```

- Run the model

```
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

# The predict method of sklearn's RandomForestClassifier returns a binary classification (0 or 1). 
# The following code creates a wrapper function, SklearnModelWrapper, that uses 
# the predict_proba method to return the probability that the observation belongs to each class. 

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]

experiment_name = "/Shared/wine_experiment/"
mlflow.set_experiment(experiment_name)
# mlflow.start_run creates a new MLflow run to track the performance of this model. 
# Within the context, you call mlflow.log_param to keep track of the parameters used, and
# mlflow.log_metric to record metrics like accuracy.
with mlflow.start_run(run_name='untuned_random_forest'):
  n_estimators = 10
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
  model.fit(X_train, y_train)

  # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
  predictions_test = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, predictions_test)
  mlflow.log_param('n_estimators', n_estimators)
  # Use the area under the ROC curve as a metric.
  mlflow.log_metric('auc', auc_score)
  wrappedModel = SklearnModelWrapper(model)
  # Log the model with a signature that defines the schema of the model's inputs and outputs. 
  # When the model is deployed, this signature will be used to validate inputs.
  signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
  
  # MLflow contains utilities to create a conda environment used to serve models.
  # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
  conda_env =  _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
        additional_conda_channels=None,
    )
  mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)
```

- do feature importance

```
feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)
```

- Get run id

```
run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id
```

- register model

```
# If you see the error "PERMISSION_DENIED: User does not have any permission level assigned to the registered model", 
# the cause may be that a model already exists with the name "wine_quality". Try using a different name.
model_name = "wine_quality"
model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)
```

- enable mlflow

```
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production",
)
```

- validate the model

```
model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# Sanity-check: This should match the AUC logged by MLflow
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')
```

- run hyper tuning

```
from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
import mlflow.xgboost
import numpy as np
import xgboost as xgb
```

- enable mlflow

```
params = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
mlflow.xgboost.autolog()
with mlflow.start_run(nested=True):
  train = xgb.DMatrix(data=X_train, label=y_train)
  test = xgb.DMatrix(data=X_test, label=y_test)
  # Pass in the test set so xgb can track an evaluation metric. XGBoost terminates training when the evaluation metric
  # is no longer improving.
  booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                      evals=[(test, "test")], early_stopping_rounds=50)
  predictions_test = booster.predict(test)
  auc_score = roc_auc_score(y_test, predictions_test)
  mlflow.log_metric('auc', auc_score)

  signature = infer_signature(X_train, booster.predict(train))
  mlflow.xgboost.log_model(booster, "model", signature=signature)
```

- stop the mlflow tracking

```
mlflow.end_run()
```

- predict the model

```
model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')
```

- Batch score or batch inference

```
# To simulate a new corpus of data, save the existing X_train data to a Delta table. 
# In the real world, this would be a new batch of data.
spark_df = spark.createDataFrame(X_train)
# Replace <username> with your username before running this cell.
table_path = "dbfs:/mlflowdata/delta/wine_data"
# Delete the contents of this path in case this cell has already been run
dbutils.fs.rm(table_path, True)
spark_df.write.format("delta").save(table_path)
```

```
import mlflow.pyfunc

apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/production")
```

```
# Read the "new data" from Delta
new_data = spark.read.format("delta").load(table_path)
```

```
display(new_data)
```

```
from pyspark.sql.functions import struct

# Apply the model to the new data
udf_inputs = struct(*(X_train.columns.tolist()))

new_data = new_data.withColumn(
  "prediction",
  apply_model_udf(udf_inputs)
)
```

```
display(new_data)
```

- Write back to delta storage

```
new_data.write.format("delta").mode("overwrite").save("/mnt/mlflow/wineoutput")
df1 = spark.read.format("delta").load("/mnt/mlflow/wineoutput")
display(df1)
```

## Azure DevOps

- go to DevOps
- Create a project
- Create a new build
- Connect to the repo
- Create a new pipeline as azure-pipelines.yml
- Sample code available in this repo - https://github.com/balakreshnan/sparkops
- First create variables

```
clusterid
databricks_token
```

- Cluster is obtained from Azure databricks workspace using JSON output
- databricks_token is personal access token
- DevOps build pipeline code

```
trigger:
- main

pool:
  vmImage: ubuntu-latest

steps:
- script: echo Hello, world!
  displayName: 'Run a one-line script'

- script: |
    echo Add other tasks to build, test, and deploy your project.
    echo See https://aka.ms/yaml
  displayName: 'Run a multi-line script'

- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.7'
    addToPath: true
    architecture: 'x64'

- task: CopyFiles@2
  inputs:
    SourceFolder: 'notebooks'
    Contents: '**'
    TargetFolder: '$(Build.SourcesDirectory)'

- task: DownloadGitHubRelease@0
  inputs:
    connection: 'balakreshnan'
    userRepository: 'balakreshnan/sparkops'
    defaultVersionType: 'latest'
    downloadPath: '$(System.ArtifactsDirectory)'

- task: Bash@3
  inputs:
    targetType: 'inline'
    script: |
      # Write your commands here
      
      ls -l $(System.ArtifactsDirectory)

- task: configuredatabricks@0
  inputs:
    url: 'https://adb-xxxxxxxxxxx.x.azuredatabricks.net'
    token: '$(databricks_token)'

- task: startcluster@0
  inputs:
    clusterid: '$(clusterid)'

- task: executenotebook@0
  inputs:
    notebookPath: '/Users/babal@microsoft.com/ML/xgboost-python'
    existingClusterId: '$(clusterid)'

- task: executenotebook@0
  inputs:
    notebookPath: '/Users/babal@microsoft.com/ML/pytorch-single-node'
    existingClusterId: '$(clusterid)'


- task: executenotebook@0
  inputs:
    notebookPath: '/Users/babal@microsoft.com/ML/mlflowexp'
    existingClusterId: '$(clusterid)'
```

- Save and queue

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/adbbuild3.jpg "Service Health")

- Details

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/adbbuild4.jpg "Service Health")

- Tasks needed

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/adbbuild1.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/adbbuild2.jpg "Service Health")

- Check Azure databricks to see if the notebooks succeded