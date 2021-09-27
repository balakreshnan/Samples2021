# How to back up using WASB driver with databricks

## Azure Databricks

## Requirements

- Back up data for Parquet, csv, json and delta
- Backup to another region
- Preserve format as WASB driver is used
- Ability to schedule
- Ability to move container after container
- Ability to move delta table

## Code

- First create a storage account with ADLS gen2 HNS enabled
- Create a Backup one as well
- Copy nyctaxi data from open source repo to first ADLS gen2 and save as delta
- Once you have delta stored let's do the coding
- Now we need to create source and destination ADLS gen2 SAS key
- Go to corresponding storage blade and create SAS key with expiration
- Save the query string with SAS some where to use in code
- Now go to Azure databricks
- create a new python notebook
- Assign a cluster with 3 nodes with auto scale disabled

- Source configuration

```
spark.conf.set(
  "fs.azure.sas.containername.storagename.blob.core.windows.net",
  "?xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
```

- Read the delta dataset to validate

```
df = spark.read.format("delta").load("wasbs://containername@storageacoount.blob.core.windows.net/nyctaxiyellowdelta/")
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/backup1.jpg "Service Health")

- Now configure the destination storage settings for SAS access

```
spark.conf.set(
  "fs.azure.sas.containername.storagename.blob.core.windows.net",
  "?xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
```

- now let's create source table to copy from

```
%sql

CREATE TABLE nyctaxiyellowsource
  USING DELTA
  LOCATION 'wasbs://containername@storageaccountname.blob.core.windows.net/nyctaxiyellowdelta/'
```

- the below is only if you want to reprocess you can truncate table

```
%sql
TRUNCATE TABLE nyctaxiyellowdestination
```

- Now enable deep copy to back up delta table

```
%sql
CREATE OR REPLACE TABLE nyctaxiyellowdestination
DEEP CLONE nyctaxiyellowsource
LOCATION 'wasbs://containername@Storageaccount.blob.core.windows.net/nyctaxiyellowdelta'
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/backup2.jpg "Service Health")

- Validation Steps to count the data
- Do count on source and destination to match

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/backup3.jpg "Service Health")