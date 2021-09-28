# Process Covid vaccine proof with AI and validate

## Using AI to process Covid vaccine card output and validate

## Note

- All the data set here is from Google images
- All publicly available dataset
- There is no PII data (all Fake)

## Architecture

- End to End process using Azure Form recognizer and Custom Vision cognitive Services using Logic apps

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/covidvaccine/images/vaccinecardarch.jpg "Service Health")

## Steps

- Before we can create logic app we need to build 2 cognitive services model

## Azure Form recognizer - Extract Covid vaccine information

- Create a storage account
- Upload the images
- for the container get a SAS URL with expiration
- Go to https://fott-2-1.azurewebsites.net/ - based on new version, URL might change
- Create a connection to Blob using SAS URL and create a project

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/covidvaccine/images/cv1.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/covidvaccine/images/cv2.jpg "Service Health")

- Create Tags
- Assign the appropriate values to tag
- Go to Model and Train the model with name

## Azure Custom Vision - Extract CDC logo

- Create a new Custom vision service
- Upload minimum 15 images

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/covidvaccine/images/cv3.jpg "Service Health")

- All images used are open source fake data images from google
- Draw bounding boxes and tag then with name
- Now click Train

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/covidvaccine/images/cv4.jpg "Service Health")

- Now click Train and see the model performance
- Will take few minutes to complete the model training

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/covidvaccine/images/cv5.jpg "Service Health")

- Do a quick test

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/covidvaccine/images/cv6.jpg "Service Health")

## Create Flow with Logic App to automate processing

- Once we have the above 2 model ready then we can now automate the processing end to end
- I choose logic app to show the process
- This is not the end complete solution

- Entire flow

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/covidvaccine/images/cv7.jpg "Service Health")
![alt text](https://github.com/balakreshnan/Samples2021/blob/main/covidvaccine/images/cv8.jpg "Service Health")
![alt text](https://github.com/balakreshnan/Samples2021/blob/main/covidvaccine/images/cv9.jpg "Service Health")

- i am using blob SAS with expiration for security purpose
- Trigger the logic with blob trigger
- Create a variable to store form recognizer output URL - header - operation-location
- Using All HTTP rest connector
- make sure you have the prediction keys for both Form recognizer and custom vision
- For Form recognizer it's a 2 step process
- First send the request to analze
- then wait for 15 seconds
- send the analyzeresult url from previous header
- Make sure content-type and Ocp-Apim-Subscription-Key are sent

```
@{outputs('HTTP')['headers']?['Operation-Location']}
```

- the above is to get the output of HTTP header for analyze request
- Then send the blob SAS URI to custom vision prediction URL
- Get the prediction URL from customvision.ai web site
- Make sure the content-type and prediction-key are sent
- All the output are stored in blob storage as json
- we can do further processing based on requirements and use case