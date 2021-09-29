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

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/covidvaccine/images/cv10.jpg "Service Health")

- we can do further processing based on requirements and use case

- Form recognizer output

```
{"status":"succeeded","createdDateTime":"2021-09-28T12:43:04Z","lastUpdatedDateTime":"2021-09-28T12:43:08Z","analyzeResult":{"version":"2.1.0","readResults":[{"page":1,"angle":5.7989,"width":1130,"height":900,"unit":"pixel","selectionMarks":[{"boundingBox":[1037,184,1069,184,1069,221,1037,221],"confidence":0.223,"state":"unselected"}]}],"pageResults":[{"page":1,"tables":[{"rows":8,"columns":4,"cells":[{"rowIndex":0,"columnIndex":0,"rowSpan":2,"text":"Vaccine","boundingBox":[69,385,205,400,198,479,62,467],"isHeader":true},{"rowIndex":0,"columnIndex":1,"text":"Product Name/Manufacturer","boundingBox":[205,400,553,435,550,475,202,440],"isHeader":true},{"rowIndex":0,"columnIndex":2,"rowSpan":2,"text":"Date","boundingBox":[553,435,747,455,742,534,547,514],"isHeader":true},{"rowIndex":0,"columnIndex":3,"rowSpan":2,"text":"Healthcare Professional or Clinic Site","boundingBox":[747,455,1052,485,1048,564,742,534],"isHeader":true},{"rowIndex":1,"columnIndex":1,"text":"Lot Number","boundingBox":[202,440,550,475,547,514,198,479],"isHeader":true},{"rowIndex":2,"columnIndex":0,"rowSpan":2,"text":"1\" Dose COVID-19","boundingBox":[62,467,198,479,190,560,54,547],"isHeader":false},{"rowIndex":2,"columnIndex":1,"text":"","boundingBox":[198,479,547,514,545,553,195,519],"isHeader":false},{"rowIndex":2,"columnIndex":2,"text":"","boundingBox":[547,514,742,534,739,572,545,553],"isHeader":false},{"rowIndex":2,"columnIndex":3,"rowSpan":2,"text":"","boundingBox":[742,534,1048,564,1043,643,736,613],"isHeader":false},{"rowIndex":3,"columnIndex":1,"text":"","boundingBox":[195,519,545,553,541,594,190,560],"isHeader":false},{"rowIndex":3,"columnIndex":2,"text":"mm dd yy","boundingBox":[545,553,739,572,736,613,541,594],"isHeader":false},{"rowIndex":4,"columnIndex":0,"text":"2nd Dose","boundingBox":[54,547,190,560,187,600,51,588],"isHeader":false},{"rowIndex":4,"columnIndex":1,"text":"","boundingBox":[190,560,541,594,538,633,187,600],"isHeader":false},{"rowIndex":4,"columnIndex":2,"text":"","boundingBox":[541,594,736,613,733,652,538,633],"isHeader":false},{"rowIndex":4,"columnIndex":3,"text":"","boundingBox":[736,613,1043,643,1041,681,733,652],"isHeader":false},{"rowIndex":5,"columnIndex":0,"text":"COVID-19","boundingBox":[51,588,187,600,183,640,47,627],"isHeader":false},{"rowIndex":5,"columnIndex":1,"text":"","boundingBox":[187,600,538,633,535,674,183,640],"isHeader":false},{"rowIndex":5,"columnIndex":2,"text":"mm dd yy","boundingBox":[538,633,733,652,730,694,535,674],"isHeader":false},{"rowIndex":5,"columnIndex":3,"text":"","boundingBox":[733,652,1041,681,1039,723,730,694],"isHeader":false},{"rowIndex":6,"columnIndex":0,"text":"Other","boundingBox":[47,627,183,640,178,707,42,693],"isHeader":false},{"rowIndex":6,"columnIndex":1,"text":"","boundingBox":[183,640,535,674,530,740,178,707],"isHeader":false},{"rowIndex":6,"columnIndex":2,"text":"mm dd yy","boundingBox":[535,674,730,694,726,759,530,740],"isHeader":false},{"rowIndex":6,"columnIndex":3,"text":"","boundingBox":[730,694,1039,723,1036,790,726,759],"isHeader":false},{"rowIndex":7,"columnIndex":0,"text":"Other","boundingBox":[42,693,178,707,171,777,35,765],"isHeader":false},{"rowIndex":7,"columnIndex":1,"text":"","boundingBox":[178,707,530,740,525,811,171,777],"isHeader":false},{"rowIndex":7,"columnIndex":2,"text":"mm dd yy","boundingBox":[530,740,726,759,721,821,525,811],"isHeader":false},{"rowIndex":7,"columnIndex":3,"text":"","boundingBox":[726,759,1036,790,1034,821,721,821],"isHeader":false}],"boundingBox":[73,379,1067,450,1029,821,34,748]}]}],"documentResults":[{"docType":"custom:vacinnecard1","modelId":"3dea3bee-1230-47ed-a8d4-7240c6cf36ff","pageRange":[1,1],"fields":{"Firstname":{"type":"string","confidence":0.943},"vacineserial1":{"type":"string","confidence":0.919},"Lastname":{"type":"string","confidence":0.919},"vaccinename1":{"type":"string","confidence":0.919},"vacinedate1":{"type":"string","confidence":0.919},"vaccinename":{"type":"string","valueString":"Lot Number","text":"Lot Number","page":1,"boundingBox":[210.0,450.0,351.0,450.0,351.0,487.0,210.0,487.0],"confidence":0.99},"healthcarename":{"type":"string","confidence":0.97},"date":{"type":"string","confidence":0.974},"healthcareid":{"type":"string","confidence":0.974},"vacineserial":{"type":"string","confidence":0.973}},"docTypeConfidence":0.589}],"errors":[]}}
```

- CDC logo output

```
{"id":"8380b276-314d-40c5-9b04-c8564ca363dd","project":"93aa72a0-1f37-4097-8f95-d9002a14f4c5","iteration":"5903b0e8-9706-4443-a541-4912015208c0","created":"2021-09-28T12:43:23.684Z","predictions":[{"probability":0.9998611,"tagId":"68b532c0-6de5-4aba-8074-8f234099d1e7","tagName":"CDClogo","boundingBox":{"left":0.7381513,"top":0.12175512,"width":0.22355062,"height":0.18526697}},{"probability":0.44562727,"tagId":"68b532c0-6de5-4aba-8074-8f234099d1e7","tagName":"CDClogo","boundingBox":{"left":0.62779665,"top":0.120928116,"width":0.37220335,"height":0.1991472}},{"probability":0.011576932,"tagId":"68b532c0-6de5-4aba-8074-8f234099d1e7","tagName":"CDClogo","boundingBox":{"left":0.69671506,"top":0.044883475,"width":0.30187488,"height":0.33183825}}]}
```