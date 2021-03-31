# Bulk move data from Google GCP Bigquery

## Use case

- Move tables from GCP bigquery
- One time move

## Prerequisties

- Azure account
- Create Azure ADLS gen2
- Create Azure Data Factory
- Create and install Self hosted integration runtime.
- Integration runtime can run in windows OS VM or local computer
- Create Azure Keyvault
- GCP Account
- GCP Project
- GCP service email
- GCP permission to connect and query specific tables
- .p12 file with authentication

## Steps in Azure Data Factory

- Create a linked services
- Need Service Email for GCP bigquery access
- Need p12 security file
- Create a linked service and valdiate and test the connection

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/gcpbigquery0-1.JPG "Service Health")

- Create a Copy activity
- Create the source
- source select the GCP bigquery as dataset and use the above linked services

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/gcpbigquery1.jpg "Service Health")

- The below is the picture of data view for data set to see if you review the actual data from bigquery

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/gcpbigquery2.jpg "Service Health")

- Here is the Copy activity will look like

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/gcpbigquery3.jpg "Service Health")

- Create a Sink
- Destination on ADLS gen2 storage
- Save as csv file for now (can be parquet)

- destination linked services

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/gcpbigquery4.jpg "Service Health")

- Sink service config

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/gcpbigquery5.jpg "Service Health")

Thank you