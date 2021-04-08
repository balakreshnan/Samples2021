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
- 