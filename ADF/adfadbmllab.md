# Azure Datafactory and Azure databricks Demo/Lab

## Build a end to end data science pipeline lab using Azure data factory and azure databricks

## Pre requisties

- Azure account
- Azure data factory
- Azure databricks
- Azure Storage ADLS gen2 - to store all the parquet file - data lake
- Azure Keyvault for storing secrets

## End to End Pipeline Architecture

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfadb1.jpg "Service Health")

## Steps

- For the notebooks i am using existing notebooks from microsoft doc web site
- the business logic here has not real business value
- each tasks are not related to each other, it's to show the flow
- data used are public data sets

### Components

- Data flow to show case Data warehouse - Facts/Dimension model
- Notebook - PassingParameters - to show case how to pass pipeline parameters into notebooks
- Notebook - DataEngineering - to show case data engineering using spark
- Notebook - Machine learning MLLib - show case machine learning using spark ML Library
- Notebook - Tensorflow-Keras - show case machine learning using keras with tensorflow backend
- Notebook - Tensorflow-Distributed - show case tensordflow using horovids to distribute compute

### Data Flow

- Lets create a new data flow
- https://github.com/balakreshnan/Samples2021/blob/main/ADF/adfmultijoin.md
- Follow the above link to create the multi join data flow
- Simulates HiveQL query

### Notebooks - Azure Databricks

#### Notebook - PassingParameters

- Log into Azure Databricks
- Create a cluster with default configuration
- create a folder called adftutorial
- Create a python notebook called mynotebook
- in the first cell write the below code

```
dbutils.widgets.text("input", "","")
y = dbutils.widgets.get("input")
print ("Param -\'input':")
print (y)
```

#### Notebook - DataEngineering

- Create a folder called Demo
- Import the sample scala notebook from
- https://docs.microsoft.com/en-us/azure/databricks/_static/notebooks/delta/quickstart-scala.html
- Click import notebook and copy the URL
- Go to Azure databricks and navigate to demo folder
- Click import and paste the url
- https://docs.microsoft.com/en-us/azure/databricks/_static/notebooks/delta/quickstart-scala.html

#### Notebook - Machine Learning Spark - MLLib

- Goto folder called Demo
- Import the sample scala notebook from
- https://docs.microsoft.com/en-us/azure/databricks/_static/notebooks/getting-started/get-started-with-mllib-dbr7.html
- Click import notebook and copy the URL
- Go to Azure databricks and navigate to demo folder
- Click import and paste the url
- https://docs.microsoft.com/en-us/azure/databricks/_static/notebooks/getting-started/get-started-with-mllib-dbr7.html

#### Notebook - Tensorflow-keras

- Goto folder demo
- Create a new notebook in demo folder
- Name the notebook as: tensorflow-keras
- select python as language of choice
- Code below
- each code block in new cell

```
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import mlflow
import mlflow.keras
import mlflow.tensorflow
```

```
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
 
cal_housing = fetch_california_housing()
 
# Split 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                    cal_housing.target,
                                                    test_size=0.2)
```

```
from sklearn.preprocessing import StandardScaler
 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

- Define the model

```
def create_model():
  model = Sequential()
  model.add(Dense(20, input_dim=8, activation="relu"))
  model.add(Dense(20, activation="relu"))
  model.add(Dense(1, activation="linear"))
  return model
```

- Create the model

```
model = create_model()
 
model.compile(loss="mse",
              optimizer="Adam",
              metrics=["mse"])
```

- Run the training

```
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
 
# In the following lines, replace <username> with your username.
experiment_log_dir = "/dbfs/<username>/tb"
checkpoint_path = "/dbfs/<username>/keras_checkpoint_weights.ckpt"
 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_log_dir)
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor="loss", mode="min", patience=3)
 
history = model.fit(X_train, y_train, validation_split=.2, epochs=35, callbacks=[tensorboard_callback, model_checkpoint, early_stopping])
```

- Predict the model to evaluate

```
model.evaluate(X_test, y_test)
```

#### Notebook - Tendorflow-Distributed

- Goto folder called Demo
- Import the sample scala notebook from
- https://docs.microsoft.com/en-us/azure/databricks/_static/notebooks/deep-learning/spark-tensorflow-distributor.html
- Click import notebook and copy the URL
- Go to Azure databricks and navigate to demo folder
- Click import and paste the url
- https://docs.microsoft.com/en-us/azure/databricks/_static/notebooks/deep-learning/spark-tensorflow-distributor.html

### Azure Data Factory

- Log into Azure Datafactory Authour UI
- Create a Dataflow and select the multijoindf you created above
- Now Expand databricks and drag and drop Notebook
- name it "PassingParameters"
- Create a Azure data bricks connection
- I am using managed identity to connect to Azure databricks
- Permission for ADF managed identity has to be provided in Azure databricks
- needs conributor access
- Navigate to adftutorial\mynotebook
- Create a base parameter
- Name is as "name" and in value type @pipeline().parameters.name
- Now drag notebook again
- Name is "Data Engineering"
- connect to same databricks
- Navidage to Demo/Databricks Delta Quickstart (Scala)
- Now drag notebook again
- Name is "Machine Learning Spark - MLLib"
- connect to same databricks
- Navidage to Demo/Getting Started with MLlib
- Now drag notebook again
- Name is "TensorFlow-Keras"
- connect to same databricks
- Navidage to Demo/tensorflow-keras
- Now drag notebook again
- Name is "TensorFlow-Distributed"
- connect to same databricks
- Navidage to Demo/spark-tensorflow-distributor
- Connect all the above as below

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfadb1.jpg "Service Health")

- Save the pipeline
- Name the pipeline as "E2EADB"
- Click Publish
- wait for publish to complete
- Then click Add Trigger -> Trigger Now
- Go to Monitor
- Wait for Job to complete

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfadb2.jpg "Service Health")

- Once completed click the run and view details

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfadb3.jpg "Service Health")

- Click details and check and see

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfadb4.jpg "Service Health")

- For Azure databricks notebook it should show the notebook run URL
- Click and view the notebook output.


### Done