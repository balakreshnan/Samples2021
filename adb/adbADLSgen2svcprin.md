# Azure Databricks access ADLS gen2 with Service principal

## How to access ADLS gen2 using service principal in Azure Databricks

## Use case

- Access securely data in ADLS gen2 with HNS enabled

## Steps

```
val configs = Map(
  "fs.azure.account.auth.type" -> "OAuth",
  "fs.azure.account.oauth.provider.type" -> "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
  "fs.azure.account.oauth2.client.id" -> "service principal client id",
  "fs.azure.account.oauth2.client.secret" -> "Secret",
  "fs.azure.account.oauth2.client.endpoint" -> "https://login.microsoftonline.com/directory id/oauth2/token"
)
// Optionally, you can add <directory-name> to the source URI of your mount point.
```

- Mount the system

```
dbutils.fs.mount(source = "abfss://containername@storagename.dfs.core.windows.net/", mountPoint = "/mnt/coviddata/", extraConfigs = configs)
```

- list the files

```
dbutils.fs.ls("/mnt/coviddata")
```

- Load the data

```
val df = spark.read.format("csv").option("inferSchema", "true").option("header", "true").load("/mnt/coviddata/covid_19_data.csv")
```

```
display(df)
```

```
dbutils.fs.unmount("/mnt/coviddata")
```