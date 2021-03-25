# Expense Matching using Azure AutoML

## Using Azure Machine learning service

## Requirements

- I am using data set from this article - https://drojasug.medium.com/using-sci-kit-learn-to-categorize-personal-expenses-de07b6b385f5
- Use AutoML to map expense to free form text

## Steps

- Download the data from github link in the above article
- Create a data set in Azure Machine Learning
- File name: TrainingData2.csv
- Create a data set named mapcat

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/mapcat1.jpg "Service Health")

- Select the file

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/mapcat2.jpg "Service Health")

- Select Header row as first row

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/mapcat3.jpg "Service Health")

- Should see the columns

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/mapcat4.jpg "Service Health")

- Validate and click Create

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/mapcat5.jpg "Service Health")

- Now time to create Automated ML experiment
- Create New Automated ML experiment
- Select the data set

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/mapcat6.jpg "Service Health")

- Enter the experiment name as matcapexp
- Select the column to predict as map_text
- Select the compute cluster to use

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/mapcat7.jpg "Service Health")

- Click Next
- Select classification
- Leave everything else default

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/mapcat8.jpg "Service Health")

- Click Finishe
- Wait until experiment completes

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/mapcat9.jpg "Service Health")

- Now click models to see all the model

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/mapcat10.jpg "Service Health")

- Let's check the model performance

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/mapcat11.jpg "Service Health")

- Feature importance

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/mapcat12.jpg "Service Health")