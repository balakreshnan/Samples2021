# Azure ADLS gen2 HNS storage logs

## Use Spark to visualize and audit $logs - audit logs folder

## Code

- Access the logs

```
%%pyspark
df = spark.read.load('abfss://$logs@storageaccountname.dfs.core.windows.net/blob/2021/*/*/*/*.log', format='csv'
## If header exists uncomment line below
#, header=True
)
display(df.limit(10))
```

- set the column names

```
columns = ['column1','column2','column3','column4','column5','column6', 'column7']
```

- Assign the column name

```
from pyspark.sql.functions import col

 ##columns is list from hack chat
 ##df_orig is a spark dataframe read from orig csv

col_rename = {f'_c{i}':columns[i] for i in range(0,len(columns))}
df_with_col_renamed = df.select([col(c).alias(col_rename.get(c,c)) for c in df.columns])
display(df_with_col_renamed)
```

- This columns has the data

```
df_with_col_renamed[['column1']].head()
```

- sample output

```
Row(column1='2.0;2021-01-09T14:34:07.0000203Z;GetPathAccessControl;OAuthSuccess;200;23;23;bearer;storageaccountname;storageaccountname;blob;"https://storageaccountname.dfs.core.windows.net/synapseroot/?upn=false&amp;action=getAccessControl&amp;timeout=90";"/storageaccountname/synapseroot";3e3b3b2a-901f-0022-4f94-e61a29000000;0;10.31.140.124;2018-11-09;1984;0;270;0;0;;;;Thursday')
```

- split the column

```
from pyspark.sql import functions as F
df2 = df_with_col_renamed.select(F.split('column1', ';').alias('column1'))
```

```
# If you don't know the number of columns:
df_sizes = df2.select(F.size('column1').alias('column1'))
df_max = df_sizes.agg(F.max('column1'))
nb_columns = df_max.collect()[0][0]

df_result = df2.select('column1', *[df2['column1'][i] for i in range(nb_columns)])
df_result.show()
```

- Now aggreagate based on operations

```
display(df_result.groupBy('column1[2]').count())
```
