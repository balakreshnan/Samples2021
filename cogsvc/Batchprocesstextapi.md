# Process Azure Cognitive Services API using Data Factory

## Batch process only

## Steps

- Create a Azure data factory
- Get table name and credentials for Azure SQL
- Azure SQL will be the data source input
- Output will also be saved in Azure SQL
- Create a new pipeline
- Get lookup activity
- Create a select query

```
SELECT TOP (1000) * FROM [dbo].[tblconnstr]
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/cogsvc/images/img1.jpg "Service Health")

- Now bring the foreach
- Configure the items

```
@activity('Lookup1').output.value
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/cogsvc/images/img2.jpg "Service Health")

- Now we can activity for each
- Configure the web activity to access cognitive services and get results back

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/cogsvc/images/img3.jpg "Service Health")

- Process the output and store in Azure SQL using Strored procedure activity
- Process actual data as parameter input to stored procedure

```
@string(activity('Web1').output.documents)
@item().id
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/cogsvc/images/img4.jpg "Service Health")

- Run debug and see if works

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/cogsvc/images/img5.jpg "Service Health")

- Now check the db to see if the text analytics output is stored in Azure SQL DB

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/cogsvc/images/img6.jpg "Service Health")