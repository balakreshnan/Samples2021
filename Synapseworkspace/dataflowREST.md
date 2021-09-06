# Azure Synapse analytics Data Flow data processing with REST as Sink

## Data Flow data processing sink as REST

## Note

- This article is to show the functionality

## Prerequistie

- Azure Account
- Azure synapse analytics workspace
- Azure logic app
- Azure Storage
- Upload the movidedb1.csv from this report in data folder to folder called movidedincoming
- create a new folder called moviesoutput

## Create Logic app flow

- Create a new logic app
- Create a workflow

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflowrest15.jpg "Service Health")

- Create a http trigger
- Select when Http request is received
- First delete the blob
- Save the body of request as file

```
containername: output
filename: output.json
```

## Data Flow Activity

- Log into Azure synapse analytics workspace or studio
- First upload the data

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflowrest1.jpg "Service Health")

- Create a source as moviesdb.csv file

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflowrest2.jpg "Service Health")

- Create a new linked service as csv like below

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflowrest3.jpg "Service Health")

- Bring Select now

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflowrest4.jpg "Service Health")

- Now drag filter

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflowrest5.jpg "Service Health")

```
YEAR==1960 || YEAR==1988 || YEAR==1950
```

- Derived column - create a new column

```
Rating = iif(YEAR==1998,1, toInteger(Rating))
YEAR = iif(YEAR==1960, 2021, toInteger(YEAR))
movies = iif(YEAR==1998,toInteger(movies)+1000, toInteger(movies))
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflowrest6.jpg "Service Health")

- now bring alter activity to create CRUD

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflowrest7.jpg "Service Health")

```
insert if YEAR==2021
update if YEAR==1998
delete if YEAR==1950
```

- Now sink to a REST connector

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflowrest8.jpg "Service Health")

- sink configuration

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflowrest9.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflowrest10.jpg "Service Health")

- now configure the REST setting

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflowrest11.jpg "Service Health")

- Create a pipeline and call the data flow and click debug

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflowrest12.jpg "Service Health")

- Go to metrics

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflowrest13.jpg "Service Health")

- Check the logic app logs

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflowrest14.jpg "Service Health")

- here is the logic app flow simple one

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/dataflowrest15.jpg "Service Health")

- Go to storage and see the output

```
{"movies":72104,"Title":"Balance","genresgenregenre":"Animation|Drama|Mystery|Sci-Fi|Thriller","YEAR":1988,"Rating":4}
```