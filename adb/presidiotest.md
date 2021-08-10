# Anonymize PII Data in a dataset in spark

## Using Azure Databricks anonymization of Text with PII

## Use Case

- Ability to Anonymize PII data in a dataset
- Used for Data engineering
- used for Machine learning

## Pre Requistie

- Azure Account
- Azure Storage account
- Azure databricks
- install libraries

```
presidio-analyzer
presidio-anonymizer
```

## Reference

- PII Entities supported - https://microsoft.github.io/presidio/supported_entities/
- Repo for Presidio - https://github.com/microsoft/presidio/tree/main/docs/samples/deployments/spark

## Code in Spark

- Confirm the above presidio libraries are installed

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/presidio1.jpg "Service Health")

- Now lets write the code
- Bring all the imports

```
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities.engine import OperatorConfig
from pyspark.sql.types import StringType
from pyspark.sql.functions import input_file_name, regexp_replace
from pyspark.sql.functions import col, pandas_udf
import pandas as pd
import os
```

- Load the sample Titanic data set

```
df = spark.sql("Select * from default.titanictbl")
```

- Now display the data

```
display(df)
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/presidio2.jpg "Service Health")

- Initialize the analyizer

```
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
broadcasted_analyzer = sc.broadcast(analyzer)
broadcasted_anonymizer = sc.broadcast(anonymizer)
```

- Create the UDF to anonymize

```
def anonymize_text(text: str) -> str:
    analyzer = broadcasted_analyzer.value
    anonymizer = broadcasted_anonymizer.value
    analyzer_results = analyzer.analyze(text=text, language="en")
    anonymized_results = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<ANONYMIZED>"})},
    )
    return anonymized_results.text


def anonymize_series(s: pd.Series) -> pd.Series:
    return s.apply(anonymize_text)


# define a the function as pandas UDF
anonymize = pandas_udf(anonymize_series, returnType=StringType())
```

- Set the anonymization column name
- Will only anonymize the entities set in the link - https://microsoft.github.io/presidio/supported_entities/

```
anonymized_column = "Name"
```

- Now anonymize the data

```
# apply the udf
anonymized_df = df.withColumn(
    anonymized_column, anonymize(col(anonymized_column))
)
display(anonymized_df)
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/presidio3.jpg "Service Health")