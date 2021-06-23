# Azure Synapse Analytics - Automated Machine learning using Azure Machine learning

## Run Automated ML in Azure Syanpse Analytics Workspace

## Code

- First create a directory called titanic
- Upload the Titanic.csv file
- Available in this repo under data
- Go to Data tab and go to titanic folder
- Right click the Titanic.csv and select create spark table
- Here is the code

```
%%pyspark
df = spark.read.load('abfss://containername@storagename.dfs.core.windows.net/titanic/Titanic.csv', format='csv'
## If header exists uncomment line below
, header=True
)
df.write.mode("overwrite").saveAsTable("default.titanic")
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/automl1.jpg "Service Health")

- run the code by selecting spark version
- then go to data tab and check the database section and expand spark default database
- should see the titanic table there

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/automl2.jpg "Service Health")

- Go to data in data explorer or data hub and expand the default database and click on table titanic

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/automl3.jpg "Service Health")

- Click Enrich with new Machine learning model

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/automl4.jpg "Service Health")

- Select the Classification

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/automl5.jpg "Service Health")\

- Finalize the automated machine learning settings

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/automl6.jpg "Service Health")

- Click Create Run
- Wait for model to complete
- Running status

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/automl7.jpg "Service Health")