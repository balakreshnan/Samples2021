# Azure Cognitive Services Computer Vision and Custom Vision

## Use Computer vision to identify objects in an image

## Use Case

- Indentify objects in an image which are common in a range of scenarios
- Check for condition on objects detected For Example if certain objects are present in an image
- Use Custom vision to detect more custom objects
- use other congitive services to infuse AI into the process
- Event based

## Architecture

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision1.jpg "Service Health")

## Logic Flow

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision2.jpg "Service Health")
![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision3.jpg "Service Health")
![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision4.jpg "Service Health")
![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision5.jpg "Service Health")

## Code

- Configure Azure Storage

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision6.jpg "Service Health")

- Read the image file

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision7.jpg "Service Health")

- Now sent to Analyze image

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision8.jpg "Service Health")

- Describe Image

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision9.jpg "Service Health")

- Detect objects

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision10.jpg "Service Health")

- now parse JSON
- Then bring in condition

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision11.jpg "Service Health")

- Specifiy the IF statement condition

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision12.jpg "Service Health")

- if true send to custom vision - custom model to detect more objects

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision13.jpg "Service Health")

- Delete blob 

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision14.jpg "Service Health")

- Save blob for describe

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision15.jpg "Service Health")

- Delete Object detection

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision16.jpg "Service Health")

- Save object detection details

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision17.jpg "Service Health")

- Send to custom vision for more object

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision18.jpg "Service Health")

- delete custom vision

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision19.jpg "Service Health")

- Save data to custom vision

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision20.jpg "Service Health")