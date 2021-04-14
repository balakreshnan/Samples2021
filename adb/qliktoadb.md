# Azure Databricks Integration with QlikView

## Components

```
Note: This tutorial/ guide is for Windows Machines 
```

## Instructions

- First let’s collect Azure Databricks related configuration as follows:
    - Select Cluster
    - Azure Databricks → Clusters (select the cluster for compute and return results to QlikView)
    - Get Configuration
    - Configuration tab → Advanced Settings → JDBC/ODBC
    - Gather following highlighted details

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/clusterConfig.png "Service Health")

    - Generate personal token in Azure Databricks with proper comment and Lifetime (0 for live indefinitely)
User Settings → Generate New Token
    - Copy this data to some file for further use

- Download Simba ODBC Driver from the link Databricks-Simba-odbc-driver or alternatively get it from this git prerequisites/SimbaODBCDriver.zip
- After extracting the downloaded file, we will have msi installer file as follows:

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/SimbaExtracted.png "Service Health")

- Install the driver, by choosing appropriate options
- Navigate to setup Azure Databricks data source in the system as follows:
Control Panel → View by (Large Icons) at top right corner → Administrative Tools

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/cp-admintools.png "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/datasources.png "Service Health")

- Click Add in User DSN/ System DSN, based on your needs.

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/create-dsn.png "Service Health")

- Make sure to select/ update with following configuration

| Parameter/ Setting                  | Value                                                    |
|-------------------------------------|----------------------------------------------------------|
| Data Source Name                    | as required                                              |
| Description                         | optional                                                 |
| Spark Server Type                   | SparkThriftServer                                        |
| Service Discovery Mode              | No Service Discovery                                     |
| Host                                | HOST_FROM_DATABRICKS_CLUSTER                             |
| Port                                | PORT_FROM_DATABRICKS_CLUSTER (default value: 443)        |
| Database                            | default is sufficient for this integration               |
| Authentication                      | OAuth 2.0                                                |
| Delegation UID                      | BLANK                                                    |
| OAuth Options > Authentication Flow | Token Passthrough                                        |
| OAuth Options > Access Token        | AAD Access token                                         |
| Thrift Transport                    | HTTP                                                     |
| SSL Options                         | Enable SSL (check this) & let it use default cacerts.pem |
| HTTP Options                        | HTTP_PATH_FROM_DATABRICKS_CLUSTER                        |

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/svc-oauthoptions.png "Service Health")

- Test Connection (this starts the cluster if it’s in terminated state and wait for the respone from cluster to the DSN ODBC driver Setup)
    - Result from DSN Setup

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/svc-dnssuccess.png "Service Health")

- Open QlikView and Edit Script (Ctrl+E)
- Select ODBC Database & Click connect and check Show USER DSNs (if you have setup USER DSN rather than System DSN) → Select Data Source name we have created → Click OK (if required, you can test connection)

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/svc-selectODBC.png "Service Health")

- Click Select to select the database and tables to be loaded into QlikView reports as follows:

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/svc-selectdataset.png "Service Health")

- Click Reload once we have the SQL statement append to the file, this queries and loads data from databricks database to the qvw table object

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/svc-appendedtofile.png "Service Health")

- Following is the data, when displayed using
    - Object → New Sheet Object → Table Box

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/adb/images/svc-result.png "Service Health")