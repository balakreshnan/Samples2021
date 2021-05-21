# Process AVRO files in Azure Synapse Analytics Interate Data Flow

## IoT and other schema changeable format

## Requirements

- Azure Account
- Azure Storage Account
- Upload sample avro or generate sample
- create a container and upload the avro file
- Find the avro schema
- Azure synapse analytics workspace
- Create a intergation runtime

## Steps

- First create a storage container
- I have uploaded 22.avro sample file in the data folder in this repo
- Sample is below

```
SequenceNumber,Offset,EnqueuedTimeUtc,SystemProperties,Properties,Body,BodyNew
507,364464,5/20/2021 11:46:23 PM,,,"{\"applicationId\":\"3a11d300-d923-4a3d-9cd9-75364b23f710\",\"messageSource\":\"telemetry\",\"deviceId\":\"mymxchipbb\",\"schema\":\"default@v1\",\"templateId\":\"urn:6bccexgax:ex1fym5he\",\"enqueuedTime\":\"2021-05-20T23:46:21.84Z\",\"telemetry\":{\"gyroscope\":{\"z\":350,\"x\":1400,\"y\":-3150},\"accelerometer\":{\"x\":-67,\"y\":-735,\"z\":686},\"humidity\":50,\"temperature\":30.299999,\"pressure\":999.015137,\"magnetometer\":{\"y\":270,\"z\":-251,\"x\":185}},\"messageProperties\":{},\"enrichments\":{}}","{\"applicationId\":\"3a11d300-d923-4a3d-9cd9-75364b23f710\",\"messageSource\":\"telemetry\",\"deviceId\":\"mymxchipbb\",\"schema\":\"default@v1\",\"templateId\":\"urn:6bccexgax:ex1fym5he\",\"enqueuedTime\":\"2021-05-20T23:46:21.84Z\",\"telemetry\":{\"gyroscope\":{\"z\":350,\"x\":1400,\"y\":-3150},\"accelerometer\":{\"x\":-67,\"y\":-735,\"z\":686},\"humidity\":50,\"temperature\":30.299999,\"pressure\":999.015137,\"magnetometer\":{\"y\":270,\"z\":-251,\"x\":185}},\"messageProperties\":{},\"enrichments\":{}}"
508,365184,5/20/2021 11:46:32 PM,,,"{\"applicationId\":\"3a11d300-d923-4a3d-9cd9-75364b23f710\",\"messageSource\":\"telemetry\",\"deviceId\":\"mymxchipbb\",\"schema\":\"default@v1\",\"templateId\":\"urn:6bccexgax:ex1fym5he\",\"enqueuedTime\":\"2021-05-20T23:46:31.997Z\",\"telemetry\":{\"accelerometer\":{\"x\":-67,\"y\":-735,\"z\":686},\"humidity\":50,\"temperature\":30.4,\"pressure\":999.022461,\"magnetometer\":{\"x\":188,\"y\":271,\"z\":-251},\"gyroscope\":{\"z\":350,\"x\":1330,\"y\":-3150}},\"messageProperties\":{},\"enrichments\":{}}","{\"applicationId\":\"3a11d300-d923-4a3d-9cd9-75364b23f710\",\"messageSource\":\"telemetry\",\"deviceId\":\"mymxchipbb\",\"schema\":\"default@v1\",\"templateId\":\"urn:6bccexgax:ex1fym5he\",\"enqueuedTime\":\"2021-05-20T23:46:31.997Z\",\"telemetry\":{\"accelerometer\":{\"x\":-67,\"y\":-735,\"z\":686},\"humidity\":50,\"temperature\":30.4,\"pressure\":999.022461,\"magnetometer\":{\"x\":188,\"y\":271,\"z\":-251},\"gyroscope\":{\"z\":350,\"x\":1330,\"y\":-3150}},\"messageProperties\":{},\"enrichments\":{}}"
```

- Usually if the Body column is base 64 we need to convert to string as BodyNew

## Azure Synapse integrate pipeline

- Go to Azure Synapse Analytics Workspace Studio
- Go to manage
- create a new integration runtime with 16+ cores for spark processing
- Go to Develop
- Creaet a new data flow

- Connect to source as the storage account created with new avro file
- Create a new dataset connecting to data store

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/avrodf2.jpg "Service Health")

- Turn the debug on
- go to Data preview

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/avrodf1.jpg "Service Health")

- now add select to select columns

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/avrodf3.jpg "Service Health")

- Create a dervice column to convert Body (if base 64) convert to string with new column name BodyNew

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/avrodf4.jpg "Service Health")

- Now add Parse
- Create a new column as json
- in Expression select the column as data either Body or BodyNew

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/avrodf5.jpg "Service Health")

- for output column type

```
(applicationId as string,		messageSource as string,		deviceId as string,		schema as string,		templateId as string,		enqueuedTime as string,		telemetry as (gyroscope as (z as float,		x as float,		y as float),		accelerometer as (z as float,		x as float,		y as float),		humidity as float,		temperature as float,		pressure as float,		magnetometer as (z as float,		x as float,		y as float)),		messageProperties as (messageProp as string),		enrichments as (userSpecifiedKey as string))
```

- Finally use sink and store as parquet for further processing.

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/avrodf8.jpg "Service Health")

- Take a look at data and see if you can see the parse data like below

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/avrodf6.jpg "Service Health")

- Expand the telemetry column and see if you can see the sensor details

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/avrodf7.jpg "Service Health")