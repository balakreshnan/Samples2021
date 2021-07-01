# Azure Blob Size invetory report viewer

## First configure Blob inventory preview

- Here is the document
- https://docs.microsoft.com/en-us/azure/storage/blobs/blob-inventory

## Code

```
SELECT LEFT([Name], CHARINDEX('/', [Name]) - 1) AS Container, 
        COUNT(*) As TotalBlobCount,
        SUM([Content-Length]) As TotalBlobSize
FROM OPENROWSET(
    bulk 'https://storageacctname.dfs.core.windows.net/containername/2021/06/17/20-28-55/DefaultRule-AllBlobs.csv',
    format='csv', parser_version='2.0', header_row=true
) AS Source
GROUP BY LEFT([Name], CHARINDEX('/', [Name]) - 1)
```

## Spark Code

```
%%pyspark
df = spark.read.load('abfss://containername@storageacctname.dfs.core.windows.net/2021/06/17/20-28-55/DefaultRule-AllBlobs.csv', format='csv'
## If header exists uncomment line below
, header=True
)
display(df.limit(10))
```

```
from pyspark.sql.types import IntegerType
data_df = df.withColumn("Content-LengthInt", df["Content-Length"].cast(IntegerType()))
```

```
import pyspark.sql.functions

split_col = pyspark.sql.functions.split(df['Name'], '/')
data_df = data_df.withColumn('Container', split_col.getItem(0))
```

```
display(data_df.groupBy("Container").sum("Content-LengthInt"))
```