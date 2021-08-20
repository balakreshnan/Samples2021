# Azure Synapse analytics Copy activity lineage in Azure Purview

## Copy activity lineage in Azure Purview

## Note

- This article is to show the functionality

## Prerequistie

- Azure Account
- Azure synapse analytics workspace
- Azure Purview
- Link both of those
- Provide Purview contributor to synapse analytics managed identity
- Azure Storage
- Upload the titanic.csv from this report in data folder to folder called titianic
- create a new folder called titanicoutput

## Copy Activity

- Create a new integrate pipeline
- Drag and drop copy activity
- For source select csv with ADLS gen2

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/copylineage1.jpg "Service Health")

- here is the configuration for source

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/copylineage2.jpg "Service Health")

- Now configure the sink as csv as well to ADLS gen2

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/copylineage3.jpg "Service Health")

- here is the configuration for sink

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/copylineage4.jpg "Service Health")

- Now commit and click Debug

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/copylineage5.jpg "Service Health")

- Now time to go to Azure Purview we created
- Click Browse assets and then Azure Synapse Analytics

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/copylineage6.jpg "Service Health")

- Select Azure Synapse Analytics and select the workspace

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/copylineage7.jpg "Service Health")

- Now expland click the pipeline
- This shows the pipeline lineage

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/copylineage8.jpg "Service Health")

- Now navigate to asset and see the lineage

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/copylineage10.jpg "Service Health")

- That's all