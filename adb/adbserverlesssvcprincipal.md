# Azure Databricks connecting to Azure synapse serverless using service principal

## Use case

- Access data from Azure synapse serverless sql using view

## Steps

## Service principal creation

- Create a service principal
- Name is "svcprincipal"

https://docs.microsoft.com/en-us/azure/azure-sql/database/authentication-aad-service-principal-tutorial

- Store the service principal details in azure keyvault

## Azure Synapse Workspace - Serverless SQL

- Go to Azure Synapse analytics workspace
- Connect to serverless

```
use master;
Create login [svcprincipal] FROM EXTERNAL PROVIDER;

GRANT CREATE ANY DATABASE to [svcprincipal];

DROP login [svcprincipal];

CREATE USER [svcprincipal] FROM EXTERNAL PROVIDER;

ALTER ROLE db_datareader ADD MEMBER [svcprincipal]
ALTER ROLE db_owner ADD MEMBER [svcprincipal]

-- DROP user [svcprincipal];

select * from dbo.region
```

## Azure databricks Spark

- Provide Blob storage reader role for service prinicpal for underlying storage as well
- Log into Azure Databricks
- Create a new cluster
- install library - com.microsoft.azure:spark-mssql-connector_2.12_3.0:1.0.0-alpha, com.microsoft.azure:adal4j:1.6.5
- Create a new scala notebook
- Pull the secrets from Key vault

```
import com.microsoft.aad.adal4j.{AuthenticationContext, ClientCredential}
```

```
import org.apache.spark.sql.SparkSession
import java.util.concurrent.Executors
```

- Now form the JDBC URL

```
val url = "jdbc:sqlserver://servername-ondemand.sql.azuresynapse.net:1433;databaseName=dbname"
val dbTable = "dbo.tablename"
```

- Bring the keyvault secrets for service principal details

```
val principalClientId = dbutils.secrets.get(scope = "allsecrects", key = "acclientid")
val principalSecret = dbutils.secrets.get(scope = "allsecrects", key = "accsecret")
val TenantId = dbutils.secrets.get(scope = "allsecrects", key = "tenantid")
```

- Set up the Azure AD auth

```
val authority = "https://login.windows.net/" + TenantId
val resourceAppIdURI = "https://database.windows.net/"
```

- Setup auth and get token

```
val service = Executors.newFixedThreadPool(1)
val context = new AuthenticationContext(authority, true, service);
val ClientCred = new ClientCredential(principalClientId, principalSecret)
val authResult = context.acquireToken(resourceAppIdURI, ClientCred, null)

val accessToken = authResult.get().getAccessToken
```

- load data from serverless sql

```
val jdbcDF = spark.read
    .format("com.microsoft.sqlserver.jdbc.spark")
    .option("url", url)
    .option("dbtable", dbTable)
    .option("accessToken", accessToken)
    .option("encrypt", "true")
    .option("hostNameInCertificate", "*.database.windows.net")
    .load()
```

- Display the dataframe

```
display(jdbcDF)
```