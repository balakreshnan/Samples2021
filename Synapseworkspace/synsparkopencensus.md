# Azure Synapse Spark - Open census to log run time logs

## Synapse Spark

## Using open census library to push error logs to Azure monitor

## Prerequistie

- Azure account
- Azure Databricks
- Azure Storage

## Steps

- Create a synapse spark cluster

## Code

- Create a spark 3 cluster
- write the below code

```
import logging

logger = logging.getLogger(__name__)
```

```
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

logger = logging.getLogger(__name__)
```

```
# TODO: replace the all-zero GUID with your instrumentation key.
logger.addHandler(AzureLogHandler(
    connection_string='InstrumentationKey=xxxxxxxxxxxxxxx')
)
```

```
logger.warning("Sample from open census test 01")
logger.error("Sample from open census test 02")
#logger.log("Sample from open census test 03")
```

- Test Code error

```
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.trace.tracer import Tracer
```

```
properties = {'custom_dimensions': {'key_1': 'value_1', 'key_2': 'value_2'}}

# Use properties in exception logs
try:
    result = 1 / 0  # generate a ZeroDivisionError
except Exception:
    logger.exception('Captured an exception.', extra=properties)
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/opencensus2.jpg "Service Health")