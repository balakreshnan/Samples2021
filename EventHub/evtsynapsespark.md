# Azure Synapse Spark with Azure Event Hubs

## Process Streaming or event driven data using Event hub into Azure Synapse Analytics Workspace

## Pre requistie

- Azure Account
- Azure Event Hub
- Azure Event Hub data generator - https://eventhubdatagenerator.azurewebsites.net/
- Azure Storage account
- Azure Synapse workspace
- Azure Keyvault

## Steps

- Create a new spark pool in Azure Synapse workspace
- GO to Azure Event hub create a new event hub called synapseincoming
- Set the parition to 1 as this is for testing
- Go to Shared access policy and create a key to write and copy the connection string
- Go to Azure Keyvault and store the key
- Go to Eventhub name space and copy the connections string
- Copy the event hub name
- The above information is used for data generator
- Now lets write the code
- Go to Azure Synapse Analytics workspace
- Go to manage and credentials
- Add the new eventhub synapseincoming connection string to credential
- We are getting the keys from keyvault stored above

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/EventHub/images/evtspark1.jpg "Service Health")

- Now lets create the code to read the events/messages from event hub and write to serverless sql table
- destination is serverless sql table
- Get the connection string securely from credentials
- Create a new notebook and select pyspark as language

## Code

- Get the connection string

```
keyVaultName = "keyvaultname";
secretName = "synapseeventhub";
```

- Read from credentials

```
secret = mssparkutils.credentials.getSecret(keyVaultName, secretName)
```

- Configure the Event hub conf with connection string

```
connectionString = secret
ehConf = {
  'eventhubs.connectionString' : sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(connectionString)
}
```

- Create a definition to write to table

```
def write2table(df2, epoch_id):
    df2.write.mode("append").saveAsTable("default.eventhubdata")
```

- now read the stream

```
df = spark \
    .readStream \
    .format("eventhubs") \
    .options(**ehConf) \
  .load()
```

- Convert the base64 String to string

```
df1 = df.withColumn("body", df["body"].cast("string"))
```

- Now time the output stream

```
df1.writeStream \
    .outputMode("update") \
    .trigger(processingTime='5 seconds') \
    .option("checkpointLocation","abfss://eventhubdata@accsynapsestorage.dfs.core.windows.net/evetcheckpoint/") \
    .foreachBatch(write2table) \
    .start() \
    .awaitTermination()
```

- Execute each cell
- Once the write stream is ran, now we are ready to send data

## Event hub Data generator

- Go to Azure Event hub data generator
- https://eventhubdatagenerator.azurewebsites.net/
- Use the Event hub name space connection string
- Type the event hub name as synapseincoming
- for event counts we can choose 500
- Click Submit

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/EventHub/images/evtspark2.jpg "Service Health")

- Wait for few secsonds.

## Azure Synapse Serverless SQL

- Go to Serverless SQL
- Create a new query
- Let's do a count

```
SELECT count(*)
 FROM [default].[dbo].[eventhubdata]
```

- Now display the latest record

```
Select top 300 * from dbo.eventhubdata order by enqueuedTime DESC;
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/EventHub/images/evtspark3.jpg "Service Health")