# Azure Databricks Streaming with GCP Pub Sub

## Stream Pub/Sub topic using Azure Databricks

## Use Case

- Multicloud data processing
- ability to move data from GCP Pub/sub to Azure databricks to ADLS gen2
- Store as delta format
- Event driven data processing

## Architecture

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/GCP/images/pubsubadb.jpg "Service Health")

## Steps

## GCP 

- Create GCP account
- Create a project
- Create pub/sub
- Create a topic
- Create authentication for service account - https://cloud.google.com/docs/authentication/getting-started
- Provide permissions to read Topic

## Azure

- Create a Azure account
- Create a Resource group
- Create a Azure databricks
- Create a Azure Storage account - ADLS gen2 (delta storage)
- Create a cluster with runtime 8.2ML
- Here is the connector URL - https://github.com/googleapis/java-pubsublite-spark
- Once cluster is started go to library and select maven

```
com.google.cloud:pubsublite-spark-sql-streaming:0.2.0
```

- Wait for cluster to install
- Meanwhile gather the GCP project id and JSON key file
- Create a Notebook with python as language
- Read stream

```
df = spark.readStream \
  .format("pubsublite") \
  .option("pubsublite.subscription", "projects/$PROJECT_NUMBER/locations/$LOCATION/subscriptions/$SUBSCRIPTION_ID") \
  .option("gcp.credentials.key", "<SERVICE_ACCOUNT_JSON_IN_BASE64>") \
  .load
```

- Now write back to Delta for further processing

```
events.writeStream
  .format("delta")
  .outputMode("append")
  .option("checkpointLocation", "/delta/events/_checkpoints/etl-from-pubsub")
  .start("/delta/pubsub")
```

- run the notebook cell and once writestream is invoked please check the folder to see if data is getting written
- Check delta/pubsub folder in ADLS gen2