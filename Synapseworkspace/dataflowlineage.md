# Azure Synapse analytics Data Flow data processing lineage in Azure Purview

## Data Flow data processing lineage in Azure Purview

## Note

- This article is to show the functionality

## Prerequistie

- Azure Account
- Azure synapse analytics workspace
- Azure Purview
- Link both of those
- Provide Purview contributor to synapse analytics managed identity
- Azure Storage
- Upload the movidedb1.csv from this report in data folder to folder called movidedincoming
- create a new folder called moviesoutput

## Data Flow Activity

- Log into Azure synapse analytics workspace or studio
- First upload the data

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow1.jpg "Service Health")

- Go to Develop and create a data flow
- Create a new dataflow
- Here is the entire flow

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow2.jpg "Service Health")

- Bring source as csv file

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow3.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow4.jpg "Service Health")

- Now we are going to bring Select to select columns needed

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow5.jpg "Service Health")

- Next we are going to do CRUD operation with delta
- So we are going to filter few years from the data set

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow6.jpg "Service Health")

```
YEAR==1960 || YEAR==1988 || YEAR==1950
```

- Now we are going to apply some conditions and dervice columns

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow7.jpg "Service Health")

- for Rating

```
iif(YEAR==1998,1, toInteger(Rating))
```

- for YEAR

```
iif(YEAR==1960, 2021, toInteger(YEAR))
```

- For Movies

```
iif(YEAR==1998,toInteger(movies)+1000, toInteger(movies))
```

- Now bring Alter row to qualify for CRUD operation with delta

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow8.jpg "Service Health")

- apply the formula as seen in the above picture

```
YEAR==2021
YEAR==1998
YEAR==1950
```

- Now sink to delta location with insert, update and delete

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow9.jpg "Service Health")

- Configure the setting

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow10.jpg "Service Health")

- Select YEAR, and movies as the key to perform the CRUD operation
- Now create a new integrate pipeline
- Drag and drop dataflow activity and select the data flow create above

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow11.jpg "Service Health")

- Now run the pipeline in debug mode
- Give few minutes to start teh debug cluster and run the data processing above
- See the output
- IN progress

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow12.jpg "Service Health")

- Wait until it completes

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow13.jpg "Service Health")

- Check the output

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow14.jpg "Service Health")

- Check Sink activity

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow15.jpg "Service Health")

- Now check the delta output

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow16.jpg "Service Health")

- Now go to serverless sql query and delta table to view the output and validate
- i am going to check 2021 since that didn't exist in the actual data

```
SELECT TOP 10 *
FROM OPENROWSET(
    BULK 'https://storagename.dfs.core.windows.net/synapseroot/moviesoutgoing1/',
    FORMAT = 'delta') as rows Where YEAR = 2021;
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow17.jpg "Service Health")

- Within the workspace if you have linked purview then go to search bar

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow18.jpg "Service Health")

- Select the dataflow activity
- Click Lineage

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow19.jpg "Service Health")

## Azure Purview

- Login into Purview
- Go to Browse assets
- Select Azure synapse analytics
- Select your instance name

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow20.jpg "Service Health")

- Now select the pipeline with data flow

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflow21.jpg "Service Health")