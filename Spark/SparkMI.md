# Azure Synapse spark use managed identity to access data using Linked Service

## Steps

- First go to manage and create a new linked service called "ADLSgen2MI"
- Use Managed identity to authenticate the resource
- Provide Managed identity to access the storage
- I gave Storage data contributor
- Click test and make sure the connection is successful

## Spark Code

- Now go to Develop
- Create a new notebook
- Select the spark pool
- now use the below code

```
%%pyspark
# Python code
spark.conf.set("spark.storage.synapse.linkedServiceName", "ADLSgen2MI")
spark.conf.set("fs.azure.account.oauth.provider.type", "com.microsoft.azure.synapse.tokenlibrary.LinkedServiceBasedTokenProvider")

df = spark.read.csv('abfss://containername@storageaccountname.dfs.core.windows.net/2021/06/17/20-28-55/DefaultRule-AllBlobs.csv', header="true")

display(df)
```