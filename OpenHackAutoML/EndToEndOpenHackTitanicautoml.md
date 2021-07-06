# Do it yourself open hack for automated machine learning

## Use open source Titanic data set to run automated machine learning

## Pre-requistie

- Azure Account
- Create a resource group
- Creaet Azure Storage account
- Create Azure Machine learning Services

## Data

- Data is available in the repo
- Filename Titanic.csv
- Download the file to local hard drive for future upload

## Build Model

- Log into Azure Portal
- Go to Azure Machine learning services resource
- Open Azure Machine learning Studio

## Create Dataset

- Go to dataset
- Click Create Dataset

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml1.jpg "Service Health")

- Select from Local Files
- Give a name for data set: TitanicTraining

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml2.jpg "Service Health")

- Click next
- Upload the file to default data store

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml3.jpg "Service Health")

- Select Upload Files

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml4.jpg "Service Health")

- select the file from local hard drive

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml5.jpg "Service Health")

- Upload file and click Next

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml6.jpg "Service Health")

- Valdiate the schema and click Next 

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml7.jpg "Service Health")

- Check the data types and leave as is and click next

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml8.jpg "Service Health")

- Click Create

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml9.jpg "Service Health")

## Create Compute Cluster

- Go to Compute

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml11.jpg "Service Health")

- Click Compute clusters
- Create new compute cluster
- Here is the compute i choose:

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml12.jpg "Service Health")

- Now select the Compute

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml13.jpg "Service Health")

- Give a name

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml14.jpg "Service Health")


## Automated ML Experiment

- Create Automated ML

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml10.jpg "Service Health")

- Give a name as : TitanicOpenHackExp

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml15.jpg "Service Health")

- Give a experiment name
- Select Label column
- Select compute

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml16.jpg "Service Health")

- Select the type of Modelling classification or regression

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml17.jpg "Service Health")

- Click Next and then Finish

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml18.jpg "Service Health")

- Wait for Experiment to complete
- Usually with F16's and 4 nodes takes about 25 minutes
- Depending on data size and compute size time might vary.

## Outcomes and Logs

- Here is the experiment running screen

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/OpenHackAutoML/images/titautoml19.jpg "Service Health")

