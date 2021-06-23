# Azure Data/AI platform architecture - Analytics

## Analytical Data/AI platform reference archtiecture

## Use Case

- Data Driven organization
- Data Assets based insights
- Self Service
- Advanced analytics
- Data science
- Machine learning/Deep learning

## Archtiecture

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Architecture/images/AzureDataRefArch4.jpg "Service Health")

## Explained

- Architecture follows from left to right
- From On premise to Cloud in a secure way

## Data Source

- Onpremise data sources
- Cloud data sources
- IoT/Ind IoT devices
- AI-IoT Devices

### Ingestion

- Use integrate with intergration runtime for bringing batch data from onpremise source
- use integrate to bring batch data from other cloud based data sources
- CDC can be orchestrated
- Delta lake can be used
- Land in Raw zone
- For Streaming and Event based use Event hub
- Can be Kafka clients sending to Event hub kafka endpoint
- IoT hub used for Iot enabled devices as source
- Stream analytics can be used to persist data from Event hub or IoT Hub
- Stream analytics persist to storage - ADLS gen2

### Curation

- Integrate can be used to also do curation
- Serverless SQL can be used to take the data from raw -> processed
- Integrate as dataflow for drag and drop ELT/ETL type activity
- Common data model can be used to persist for final
- Data quality, Validation and aggregation all happens here
- Purview can capture lineage and data catalog information
- Type 1, typ2 and other data processing methodology can be applied here
- destination can be data warehouse, data lake, data swamp, data hub, data mesh or any other data destsination
- Delta processing is also available if we need to use it.
- Final data set should be ready for conumsers to use it for business
- Also able to use for Self Service data driven decision making

### Analytical/ML/AI

- To do what if anlaysis
- Find pattern in data
- Dedicated SQL pool is only for large volume data with faster response time
- Azure Machine learning for building advanced analytical model
- Apply Machine/deep/Reinforcement learning
- Provides Data Science life cycle management
- Model management is also available
- No Code/Low Code modelling
- Code based using jupyter or python is also available.
- Data sourced from data lake and writen back to data lake
- Vision/Audio based Deep learning is also possible with Azure ML
- Model inferenceing is deployed to Azure Kubernetes service (missing in arch - represented as app services)

### Visulaization and Web App

- Most data visulaztion is dashboard or reports
- Use Power BI for visulaization
- For Converstation UI use Bot Service with cognitive services API
- For RPA - robotic process automation use Power platform
- For Web/UI we can use Power Apps in power platform

### Common services for operations

- These are services cut across all azure paas services
- Used for management and operations
- Azure Purview used as Data governance tool
    - Data Catalog
    - Data compliance report for GDPR/PII and other compliance
    - Data classification
    - Automatic scanning or scheduled scanning of data sources
    - Bring lineage from Integrate/ADF
    - Business GLossary
    - Data stewards, owners and admins specified
- Azure monitor to build data Ops dashboard to see how services and applications are performing
- Azure Devops to do CI/CD between environments
- Azure Keyvault to store secured keys in keyvault like password and other api access key
- Application insights - Storing runtime logs for troubleshooting application. More application specific
- Azure AD - single authentication for all data services
- Github - Store all code repo in github