# Data Science Data engineering

## Sensor data and daa engineering

- Sensor data comes in their own frequency
- Data files arrives as separate files for each sensor
- Our job is to combine all the file
- Load all the files to one file
- Create a dataframe
- Pivot the dataframe

```
df.pivot(index='foo', columns='bar', values='baz')
```

- resample with minute or hour interval to combine all values

```
minutes=df.resample('1Min',on='Date').mean().dropna()
df2.groupby(pd.Grouper(key='time',freq='1min')).mean()
```

- Do mean or average
- Drop nulls values
- get rid of text columns or do one hot encoding
- if needed create new features from date time like day, hour etc
- do only if needed
- do historgram and box plot to identify outliers
- do noramlization if needed
- Split the data set for training and testing
- Model training
- model predict and validation with test data
- run feature importance