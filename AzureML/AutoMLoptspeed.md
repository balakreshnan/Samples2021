# How to improve Automated Machine Learning Performance

## Speed up AutoML run in CPU based model

## Use Case

- Improve model training time
- Using CPU
- Adding more nodes
- increase concurrency
- Optimize for cost
- Test regression model

## Introduction

- Choose NasaPred data from data.gov
- Remaining useful life model using regression
- Run with 2 Training Cluster configuration
    - Compute-Cluster called cpu-cluster1 with 2 node Standard_D14_v2 (16 cores, 112 GB RAM, 800 GB disk)
    - Compute-Cluster called cpu-cluster with 4 nodes Standard_F16s_v2 (16 cores, 32 GB RAM, 128 GB disk)
- Same dataset
- Same regression model
- Same default configuration
- Compute cluster details

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureML/images/amlexpauto2.jpg "Service Health")

- Computer cluster running details

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureML/images/amlexpauto4.jpg "Service Health")

- Nodes status

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureML/images/amlexpauto3.jpg "Service Health")

- Run Comparsion

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureML/images/amlexpauto1.jpg "Service Health")

- First run was 3 hours
- Second run was 23 minutes
- Default compute run with 4 nodes configuration

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureML/images/amlexpauto7.jpg "Service Health")

- Configuration for F16

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureML/images/amlexpauto6.jpg "Service Health")