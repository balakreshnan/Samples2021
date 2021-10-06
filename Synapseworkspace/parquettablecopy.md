# Parquet Table copy from one container to another

## Copy Parquet files

- Create 2 folder
- One for input and output
- Input can have individual folder with parquet files
- Output can be emtpy

## Code

- Create a list of table

```
table_list = ["covid191", "covid192"]
```

- Write copy code

```
from pyspark.sql import functions as fn
for table in table_list:
  try:
    filePath = 'copytest/input/' + table + '/'
    fileOut = 'abfss://containername@storagename.dfs.core.windows.net/copytest/output/' + table + '/'
    print(fileOut)
    df = spark.read.format("parquet").load(f"/{filePath}")
    # df.count()
    #req = spark.table(table).schmea
    #for each_field in req:
    #  df = df.withColumn(each_field.name, fn.col(each_field.name).cast(each_field.dataType))
    
    # fileout = 'copytest/input/' + fileOut + '/'
    df.repartition(1).write.mode("overwrite").parquet(fileOut)
    #df.select(req.fieldNames()).write.mode("overwrite").parquet(fileout)
    print(table, " ----->  Data loaded")
  except Exception as e:
    print(table, " -----> Data load failed", e)
    pass
```

- now wait until copy completes
- Display the results

```
from notebookutils import mssparkutils
mssparkutils.fs.ls('abfss://containername@storagename.dfs.core.windows.net/copytest/output/')
```