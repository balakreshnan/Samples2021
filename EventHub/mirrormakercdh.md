# Mirror Maker with CDH to Azure Event Hub

## Configure Kafka Mirror Maker in Cloudera to Send Events to Azure Event Hub Kafka Endpoint

## Azure Resources

- Create a Azure account
- Create Azure Event Hub
- Create a Storage account for Capture (if needed only)

## Configure Azure Event Hub

- Create a new Event hub Name Space
- Create a SAS key with manage, listen, send
- Copy the SAS connection key
- Create a Event hub with topic name

## Configure Cloudera Kafka mirror maker

- Log into Cloudera manager
- Go to Services and Select Kafka mirror makers
- For consumper please use Kafka brokers with 9092 as port
- Specify all the 3 nodes with ports

```
kafka1:9092,kafka2:9092,kafka3:9092
```

- Next go to Producter and configure
- Eventhub namespace full URL with port 9093

```
eventhubnamespace.servicebus.windows.net:9093
```

- save the configuration
- Restart the Kafka mirror makers
- Log into Azure portal
- Check the Event hub metric to see if messages are flowing.