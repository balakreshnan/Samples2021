# Parse Wired JSON in Azure Synapse analytics integrate

## Parse JSON

## JSON

- Sample JSON output

```
{"ticker_symbol":"CRM","sector":"HEALTHCARE","change":-2.85,"price":38.38}{"ticker_symbol":"QXZ","sector":"FINANCIAL","change":-1.58,"price":221.21}{"ticker_symbol":"VVY","sector":"HEALTHCARE","change":-1.47,"price":27.06}{"ticker_symbol":"QWE","sector":"TECHNOLOGY","change":-2.71,"price":200.9}{"ticker_symbol":"SED","sector":"HEALTHCARE","change":-0.01,"price":1.68}{"ticker_symbol":"TGH","sector":"FINANCIAL","change":-1.25,"price":67.21}{"ticker_symbol":"DFG","sector":"TECHNOLOGY","change":0.3,"price":175.16}{"ticker_symbol":"WFC","sector":"FINANCIAL","change":-0.48,"price":52.39}{"ticker_symbol":"NGC","sector":"HEALTHCARE","change":0.06,"price":6.89}{"ticker_symbol":"HJV","sector":"ENERGY","change":-15.42,"price":335.83}{"ticker_symbol":"ASD","sector":"FINANCIAL","change":2.39,"price":84.48}{"ticker_symbol":"QXZ","sector":"HEALTHCARE","change":1.14,"price":95.2}{"ticker_symbol":"DFT","sector":"RETAIL","change":-4.02,"price":86.71}{"ticker_symbol":"SAC","sector":"ENERGY","change":-2.39,"price":30.05}{"ticker_symbol":"MMB","sector":"ENERGY","change":-1.02,"price":18.57}{"ticker_symbol":"WFC","sector":"FINANCIAL","change":-0.39,"price":52}{"ticker_symbol":"VVS","sector":"ENERGY","change":1.57,"price":30.19}{"ticker_symbol":"DEG","sector":"ENERGY","change":-0.68,"price":7.07}{"ticker_symbol":"SAC","sector":"ENERGY","change":1.54,"price":31.59}{"ticker_symbol":"WMT","sector":"RETAIL","change":3.4,"price":85.22}{"metric_stream_name":"cs-":"Count"}
```

## Synapse Data Flow Code

- Create a new folder called input and upload the above sample data as no extension
- Create a output folder called jsonoutput
- Create a new data flow

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/wiredjson1.jpg "Service Health")

- load the file as text with custom row and column delimited
- Column delimiter as Comma
- Row delimiter as {

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/wiredjson2.jpg "Service Health")

- sample data should be like below

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/wiredjson3.jpg "Service Health")

- Now time to select the columns

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/wiredjson4.jpg "Service Health")

- Next is drag derived columns
- We need to create 4 new colums as below picture

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/wiredjson5.jpg "Service Health")

- Create new column names
- Use split and with : and take the second argument
- Convert to float and also replace } as nothing
- Now we parsed the JSON and got the necessary columns needed
- Now select the columns

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/wiredjson6.jpg "Service Health")

- Write output to parquet or csv
- i wrote to both to validate

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/wiredjson7.jpg "Service Health")

- Output to single file

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/wiredjson8.jpg "Service Health")

- Click Debug preview 

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/wiredjson9.jpg "Service Health")

- Now click integrate
- create a new pipeline
- add the data flow
- click debug and run the pipeline
- Make sure to commit the code if git is used

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/wiredjson10.jpg "Service Health")

- Check the debug output

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/wiredjson11.jpg "Service Health")

- Also check the output and execution times

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Synapseworkspace/images/wiredjson12.jpg "Service Health")