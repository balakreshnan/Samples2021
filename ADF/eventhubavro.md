# Read Event Hubs Avro messages

## Event hub Capture stores messages as Avro format with body as base 64 encoded

## Event hub regular and Event hub Kafka avro messages

- Parse Kafka messages in Event hub using kafka client
- Import Kafka libraries in Azure databricks
- Using Scala as programming lanugage

```
import kafkashaded.org.apache.kafka.common.security.plain.PlainLoginModule
```

- Now get eventhub configuration from Keyvault

```
val eventhubconnstring = dbutils.secrets.get(scope = "secretscope", key = "eventhubconnstring")
```

- Setup up kafka parameters

```
val TOPIC = "kafkaincoming"
val BOOTSTRAP_SERVERS = "eventhubnamespace.servicebus.windows.net:9093"
val EH_SASL = "kafkashaded.org.apache.kafka.common.security.plain.PlainLoginModule required username=\"$ConnectionString\" password=\"" + eventhubconnstring + "\";"
```

- read the stream

```
val df = spark.readStream
    .format("kafka")
    .option("subscribe", TOPIC)
    .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS)
    .option("kafka.sasl.mechanism", "PLAIN")
    .option("kafka.security.protocol", "SASL_SSL")
    .option("kafka.sasl.jaas.config", EH_SASL)
    .option("kafka.request.timeout.ms", "60000")
    .option("kafka.session.timeout.ms", "60000")
    .option("failOnDataLoss", "false")
    .option("startingOffsets", "earliest")
    .load()
```

- now convert the base 64 to string

```
val df1 = df.withColumn("bodynew", df("value").cast("string"))
```

- Display dataframe

```
display(df1)
```

- Check the bodynew column to see the actual text from message.