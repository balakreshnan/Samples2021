# Process Azure Cognitive Services API using Data Factory

## Batch process only

## Steps

- Create a Azure data factory
- Get table name and credentials for Azure SQL
- Azure SQL will be the data source input
- Output will also be saved in Azure SQL
- Create a new pipeline
- Get lookup activity

- in Azure SQL create a table and load sample data as below

```
Create table dbo.precribdata
(
  id int,
  ptext varchar(4000),
  inserttime datetime
)

insert into dbo.precribdata (id, ptext, inserttime) values (1, 'Subject is taking 100mg of ibuprofen twice daily.', getdate())
insert into dbo.precribdata (id, ptext, inserttime) values (2, 'Subject is taking 100mg of ibuprofen twice daily.', getdate())
insert into dbo.precribdata (id, ptext, inserttime) values (3, 'Subject is taking 100mg of ibuprofen twice daily.', getdate())
insert into dbo.precribdata (id, ptext, inserttime) values (4, 'Subject is taking 100mg of ibuprofen twice daily.', getdate())

select * from dbo.precribdata
```

- Create a select query

```
select * from dbo.precribdata
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/cogsvc/images/imgh1.jpg "Service Health")

- Now bring Foreach 

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/cogsvc/images/imgh2.jpg "Service Health")

```
@activity('Lookup1').output.value
```

- Now add Web Activity

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/cogsvc/images/imgh4.JPG "Service Health")

- Here is the URL to use below
- Add the headers as per picture above

```
https://servicename.cognitiveservices.azure.com/text/analytics/v3.1-preview.5/entities/health/jobs
```

- here is the format to send the data as Post Body

```
@concat('{
        "documents": [
            {
                "language": "en",
                "id": "',concat(item().id, concat('", "text": "', concat(item().ptext,'"
            }
        ]
    }'))))
```

- Set the variable to temp variable

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/cogsvc/images/imgh5.jpg "Service Health")

- Get the last value by spliting the text

```
@last(split(activity('Web1').output.ADFWebActivityResponseHeaders['operation-location'], '/'))
```

- Now Wait for 30 seconds

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/cogsvc/images/imgh6.jpg "Service Health")

- Now call the jobs api to get results

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/cogsvc/images/imgh7.jpg "Service Health")

```
@concat('https://servicename.api.cognitive.microsoft.com/text/analytics/v3.1-preview.5/entities/health/jobs/',variables('jobid'))
```

- write the results to Azure SQL using stored procedure

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/cogsvc/images/imgh8.jpg "Service Health")

- parameters for stored procedure

```
@item().id
@string(activity('Web2').output.results.documents)
```

- Publish all
- Click Debug

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/cogsvc/images/imgh9.jpg "Service Health")
![alt text](https://github.com/balakreshnan/Samples2021/blob/main/cogsvc/images/imgh10.jpg "Service Health")

- Go to Azure SQL and make sure data is saved for further processing

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/cogsvc/images/imgh11.jpg "Service Health")