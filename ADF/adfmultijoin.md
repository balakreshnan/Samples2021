# Azure data factory data flow multiple join

## use case how can multiple sql select join created using data flow

## Details

Here i am considering a sql select query with 5 joins. i have one fact and 4 dimensions to join

## Tables Details

- Fact Population
- Dimension State
- Dimention Countyname
- Dimenation Race
- Dimension Sex

## Steps

- Data is available in data folder
- there should be 5 folder
    - fact
    - dimcountryname
    - dimRace
    - dimsex
    - dimstate
- First create a pipeline
- Create a New data flow
- connect to fact first as source
- Select join 
- Configure state as another source
- Connect the join columns
- do the same for other dimenstions as below image.

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfdataflow6.jpg "Service Health")

- Once joins are created then save and publish
- Then go to pipeline and trigger once


## End to End Flow - dataflow

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfdataflow2.jpg "Service Health")

- Create Joins for State

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfdataflow10.jpg "Service Health")

- Create Join for County

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfdataflow11.jpg "Service Health")

- Create join for Race

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfdataflow12.jpg "Service Health")

- Create join for Sex

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfdataflow13.jpg "Service Health")

- Sink the output - Fact table

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfdataflow14.jpg "Service Health")

## Monitor output of data flow

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfdataflow3.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfdataflow4.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfdataflow5.jpg "Service Health")

## Log analytics metric output for ADF run above

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfdataflow7.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfdataflow8.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/adfdataflow9.jpg "Service Health")

- More to come