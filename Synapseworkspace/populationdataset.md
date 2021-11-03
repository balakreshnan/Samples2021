# Create Population Data Set

## Copy public data set and create Facts and dimensions

## Use Case

- Create US population data set

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/synpaseds1.jpg "Service Health")

- Go to Browse Gallery and select dataset
- Select US population by county

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/synpaseds2.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/synpaseds3.jpg "Service Health")

- To View the open source data
- Open Query in synapse workspace

```
SELECT
    TOP 100 *
FROM
    OPENROWSET(
        BULK     'https://azureopendatastorage.blob.core.windows.net/censusdatacontainer/release/us_population_county/year=*/*.parquet',
        FORMAT = 'parquet'
    ) AS [result];
```

- Now time to create facts and dimension using parquet storage
- I am using nyctazi as database. If needed create your own
- Create a datasource to write
- I already create sqlondemand Credential as per documentation

```
DROP EXTERNAL DATA SOURCE uspopulation;

use nyctaxi
CREATE EXTERNAL DATA SOURCE myuspopulation WITH (
    LOCATION = 'https://storagename.blob.core.windows.net/uspopulation', CREDENTIAL = sqlondemand
);
GO
```

- Let's create the Fact first
- parquet will be created a folder called fact/

```
use nyctaxi
CREATE EXTERNAL TABLE [dbo].[factpopulation] WITH (
        LOCATION = 'fact/',
        DATA_SOURCE = [myuspopulation],
        FILE_FORMAT = [ParquetFF]
) AS
SELECT
    *
FROM
    OPENROWSET(
        BULK     'https://azureopendatastorage.blob.core.windows.net/censusdatacontainer/release/us_population_county/year=*/*.parquet',
        FORMAT = 'parquet'
    ) AS [result];
```

- now time to create other dimensions from Fact it self to simulate join
- Create dimstate

```
CREATE EXTERNAL TABLE [dbo].[dimstate] WITH (
        LOCATION = 'dimstate/',
        DATA_SOURCE = [myuspopulation],
        FILE_FORMAT = [ParquetFF]
) AS
Select distinct StateName from dbo.factpopulation
```

- create county dim

```
CREATE EXTERNAL TABLE [dbo].[dimcountyname] WITH (
        LOCATION = 'dimcountryname/',
        DATA_SOURCE = [myuspopulation],
        FILE_FORMAT = [ParquetFF]
) AS
Select distinct CountyName from dbo.factpopulation
```

- Create race dim

```
CREATE EXTERNAL TABLE [dbo].[dimRace] WITH (
        LOCATION = 'dimRace/',
        DATA_SOURCE = [myuspopulation],
        FILE_FORMAT = [ParquetFF]
) AS
Select distinct Race from dbo.factpopulation
```

- Create sex dim

```
CREATE EXTERNAL TABLE [dbo].[dimSex] WITH (
        LOCATION = 'dimsex/',
        DATA_SOURCE = [myuspopulation],
        FILE_FORMAT = [ParquetFF]
) AS
Select distinct Sex from dbo.factpopulation
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/synpaseds4.jpg "Service Health")

- Now i was able to move the parquet using Azure data factory to another blob storage.