# Azure Google Big Query query using Azure data bricks

## Azure Databricks to access big query

## Use Case

- Access bigquery data from azure databricks for analytics
- Used for Data engineering
- used for Machine learning

## Pre Requistie

- Azure Account
- Azure Storage account
- Azure databricks
- GCP account
- GCP Project
- GCP bigquery
- Create a sample data set
- Provide permission to access and query
- Create permission JSON file and download it

## Steps

- First create a Storage account
- Create a container called gcp
- Use storage explorer to create conf folder
- upload the permission json file for GCP access
- save the file service-access.json
- Now lets go to databricks to start coding
- Configure the cluster

![alt text](https://github.com/balakreshnan/Accenture/blob/master/images/gcpbigqueryadb2.jpg "Service Health")

- Let's create notebook

```
val accbbstorekey = dbutils.secrets.get(scope = "allsecrects", key = "accbbstore")
```

```
spark.conf.set(
  "fs.azure.account.key.accbbstore.blob.core.windows.net",
  accbbstorekey)
```

```
dbutils.fs.mount(
  source = "wasbs://pmt@accbbstore.blob.core.windows.net/conf",
  mountPoint = "/mnt/gcp",
  extraConfigs = Map("fs.azure.account.key.accbbstore.blob.core.windows.net" -> dbutils.secrets.get(scope = "allsecrects", key = "accbbstore")))
```

```
dbutils.fs.ls("dbfs:/mnt/gcp")
```

- print environment variable

```
%sh printenv
```

- Configure table name

```
val table = "sbx-9403-projectname-adfsrc.stooge.stooges"
// load data from BigQuery
val df = spark.read.format("bigquery").option("table", table).load()
```

- Display the data set

```
display(df)
```

![alt text](https://github.com/balakreshnan/Accenture/blob/master/images/gcpbigqueryadb3.jpg "Service Health")

- unmount the dbfs

```
dbutils.fs.unmount("/mnt/gcp")
```