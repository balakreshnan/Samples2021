# Azure Synapse Serverless SQL running DDL commands

## Create Database and Tables and quries

## Use case

- Ability to run SQL scripts for serverless scripts
- Use ADF as CI/CD to promote serverless SQL scripts

## Pre Requistie

- Azure Account
- Azure Synapse Analytics workspace
- Azure Data factory
- Github to store Code repository

## Steps

- First Create Azure data factory
- Get the name of data factory as Managed identity
- Go to Underlying data storage for Azure Synapse
- Go to Access Control - IAM and add Data factory to Storage data reader role
- Go to Synapse workspace
- Go to Manage and add the data factory as synapse administrator
- Open SQL Query
- Create a login from managed identity

```
use master;
Create login [adfmangedidentity] FROM EXTERNAL PROVIDER;
```

- Grant access to create database

```
GRANT CREATE ANY DATABASE to [accadf1];
```

- if you want to delete then

```
DROP login [adfmangedidentity];
```

- Now go to Azure data factory
- Create a pipeline
- Drag lookup activity
- Connect using Azure synapse analytics
- Create datbase in serverless SQL

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/svrddladf1.jpg "Service Health")

- configure Azure synapse analytics connector

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/svrddladf2.jpg "Service Health")

- Configure DBname as parameter
- default value "master"

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/svrddladf3.jpg "Service Health")

- here is the details configuration

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/svrddladf4.jpg "Service Health")

- Now configure another lookup to delete the database

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/svrddladf5.jpg "Service Health")

- Let save and run
- Go to monitor and watch the output

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/svrddladf6.jpg "Service Health")