# Azure Synapse Spark logs runtime errors to Application insights

## Using open census library to push error logs to Azure monitor

## Prerequistie

- Azure account
- Azure Synapse workspace
- Azure Storage

## Steps

- Create a Spark cluster
- install library
- create a conda file and upload
- Create a environment.yml

```
name: example-environment
channels:
  - conda-forge
dependencies:
  - python
  - numpy
  - pip
  - pip:
    - opencensus-ext-azure
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/opencensus1.jpg "Service Health")

- Create a notebook

## Code

- Choose python as language

```
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

logger = logging.getLogger(__name__)
```

```
# TODO: replace the all-zero GUID with your instrumentation key.
logger.addHandler(AzureLogHandler(
    connection_string='InstrumentationKey=xxxxx-xxxxxx-xxxxxx-xxxxxxx')
)
```

- now log some sample logs

```
logger.warning("Sample from open census test 01")
logger.error("Sample from open census test 02")
```

- NOw lets log an exception

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

- log into application insights

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/opencensus3.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/opencensus4.jpg "Service Health")

- the above test for done from - https://docs.microsoft.com/en-us/azure/azure-monitor/app/opencensus-python#configure-azure-monitor-exporters