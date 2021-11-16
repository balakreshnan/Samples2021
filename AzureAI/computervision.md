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

```
{"description":{"tags":["person","woman"],"captions":[{"text":"a couple of women looking at a cell phone","confidence":0.4442202150821686}]},"requestId":"1298b6aa-2242-4425-946b-1b5632b13e46","metadata":{"height":360,"width":539,"format":"Jpeg"},"modelVersion":"2021-05-01"}
```

- Delete Object detection

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision16.jpg "Service Health")

- Save object detection details

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision17.jpg "Service Health")

```
{"objects":[{"rectangle":{"x":169,"y":223,"w":48,"h":46},"object":"cell phone","confidence":0.594,"parent":{"object":"telephone","confidence":0.595}},{"rectangle":{"x":74,"y":56,"w":213,"h":274},"object":"person","confidence":0.851},{"rectangle":{"x":232,"y":37,"w":227,"h":304},"object":"person","confidence":0.911}],"requestId":"e93a3f67-ae26-49fc-982b-e5f274fe09a6","metadata":{"height":360,"width":539,"format":"Jpeg"},"modelVersion":"2021-04-01"}
```

- Send to custom vision for more object

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision18.jpg "Service Health")

- delete custom vision

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision19.jpg "Service Health")

- Save data to custom vision

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision20.jpg "Service Health")

```
{"id":"d588fff0-5567-484b-9235-8338f227c567","project":"18fd34c4-fbff-4533-a246-a4c1038a531d","iteration":"0998c468-cf51-41f0-a215-09168637a169","created":"2021-11-16T14:29:37.808Z","predictions":[{"probability":0.4232306,"tagId":"6876629c-96a3-4fef-88af-73153acd13a0","tagName":"Person","boundingBox":{"left":0.16934127,"top":0.20376185,"width":0.39373505,"height":0.67024744}},{"probability":0.16393189,"tagId":"67e2ef59-4d5b-4127-9f09-00085ddcb919","tagName":"Shirt","boundingBox":{"left":0.14554521,"top":0.273085,"width":0.4376786,"height":0.6819979}},{"probability":0.1291166,"tagId":"6876629c-96a3-4fef-88af-73153acd13a0","tagName":"Person","boundingBox":{"left":0.34111458,"top":0.22302037,"width":0.47320133,"height":0.77576745}},{"probability":0.08303073,"tagId":"67e2ef59-4d5b-4127-9f09-00085ddcb919","tagName":"Shirt","boundingBox":{"left":0.4125951,"top":0.24784145,"width":0.4243359,"height":0.7269095}},{"probability":0.04470531,"tagId":"6876629c-96a3-4fef-88af-73153acd13a0","tagName":"Person","boundingBox":{"left":0.9532987,"top":0.42737478,"width":0.042752385,"height":0.23391199}},{"probability":0.036574155,"tagId":"d6720b77-2150-40f3-8707-d70b766a2911","tagName":"Vest","boundingBox":{"left":0.9561763,"top":0.4711132,"width":0.041060448,"height":0.2426132}},{"probability":0.03141522,"tagId":"6876629c-96a3-4fef-88af-73153acd13a0","tagName":"Person","boundingBox":{"left":0.94273466,"top":0.3834714,"width":0.057264328,"height":0.5733633}},{"probability":0.0303162,"tagId":"3bd8b25a-30b9-41d3-901f-e73664f49095","tagName":"SafetyGlass","boundingBox":{"left":0.2197346,"top":0.1902576,"width":0.2138948,"height":0.40137053}},{"probability":0.027975691,"tagId":"b5413c8e-9df4-4d9a-a2c5-dbac5caea08e","tagName":"HardHat","boundingBox":{"left":0.9561763,"top":0.4711132,"width":0.041060448,"height":0.2426132}},{"probability":0.026754478,"tagId":"d6720b77-2150-40f3-8707-d70b766a2911","tagName":"Vest","boundingBox":{"left":0.565753,"top":0.6927401,"width":0.16022152,"height":0.26027012}},{"probability":0.026746431,"tagId":"d6720b77-2150-40f3-8707-d70b766a2911","tagName":"Vest","boundingBox":{"left":0.94125813,"top":0.30639032,"width":0.058312774,"height":0.5994117}},{"probability":0.024915103,"tagId":"3bd8b25a-30b9-41d3-901f-e73664f49095","tagName":"SafetyGlass","boundingBox":{"left":0.9561763,"top":0.4711132,"width":0.041060448,"height":0.2426132}},{"probability":0.018509056,"tagId":"6876629c-96a3-4fef-88af-73153acd13a0","tagName":"Person","boundingBox":{"left":0.85062253,"top":0.46367824,"width":0.052606285,"height":0.18794453}},{"probability":0.018009502,"tagId":"b5413c8e-9df4-4d9a-a2c5-dbac5caea08e","tagName":"HardHat","boundingBox":{"left":0.37079078,"top":0.0,"width":0.21095163,"height":0.06819162}},{"probability":0.017119315,"tagId":"6876629c-96a3-4fef-88af-73153acd13a0","tagName":"Person","boundingBox":{"left":0.81569105,"top":0.4598646,"width":0.048707247,"height":0.17618498}},{"probability":0.015736777,"tagId":"d6720b77-2150-40f3-8707-d70b766a2911","tagName":"Vest","boundingBox":{"left":0.737093,"top":0.17961332,"width":0.26290601,"height":0.7084025}},{"probability":0.015264544,"tagId":"d6720b77-2150-40f3-8707-d70b766a2911","tagName":"Vest","boundingBox":{"left":0.4102777,"top":0.7318763,"width":0.44337574,"height":0.18766779}},{"probability":0.014852225,"tagId":"67e2ef59-4d5b-4127-9f09-00085ddcb919","tagName":"Shirt","boundingBox":{"left":0.9532987,"top":0.42737478,"width":0.042752385,"height":0.23391199}},{"probability":0.014332452,"tagId":"6876629c-96a3-4fef-88af-73153acd13a0","tagName":"Person","boundingBox":{"left":0.44240332,"top":0.304672,"width":0.17287946,"height":0.6026026}},{"probability":0.013621434,"tagId":"3bd8b25a-30b9-41d3-901f-e73664f49095","tagName":"SafetyGlass","boundingBox":{"left":0.116784915,"top":0.05962187,"width":0.42587817,"height":0.6842821}},{"probability":0.012613807,"tagId":"d6720b77-2150-40f3-8707-d70b766a2911","tagName":"Vest","boundingBox":{"left":0.8400014,"top":0.54358506,"width":0.13617575,"height":0.39666444}},{"probability":0.012164386,"tagId":"b5413c8e-9df4-4d9a-a2c5-dbac5caea08e","tagName":"HardHat","boundingBox":{"left":0.11138326,"top":0.0,"width":0.42763615,"height":0.7077935}},{"probability":0.011890712,"tagId":"d6720b77-2150-40f3-8707-d70b766a2911","tagName":"Vest","boundingBox":{"left":0.8496671,"top":0.48610014,"width":0.06674409,"height":0.20135003}},{"probability":0.011256634,"tagId":"6876629c-96a3-4fef-88af-73153acd13a0","tagName":"Person","boundingBox":{"left":0.2197346,"top":0.1902576,"width":0.2138948,"height":0.40137053}},{"probability":0.010340022,"tagId":"6876629c-96a3-4fef-88af-73153acd13a0","tagName":"Person","boundingBox":{"left":0.93651795,"top":0.028606087,"width":0.06348103,"height":0.71810424}},{"probability":0.0103162145,"tagId":"b5413c8e-9df4-4d9a-a2c5-dbac5caea08e","tagName":"HardHat","boundingBox":{"left":0.47340962,"top":0.0031771418,"width":0.19634786,"height":0.062218063}},{"probability":0.010020253,"tagId":"b5413c8e-9df4-4d9a-a2c5-dbac5caea08e","tagName":"HardHat","boundingBox":{"left":0.8496671,"top":0.48610014,"width":0.06674409,"height":0.20135003}},{"probability":0.009135243,"tagId":"d6720b77-2150-40f3-8707-d70b766a2911","tagName":"Vest","boundingBox":{"left":0.16485196,"top":0.5151962,"width":0.6110592,"height":0.48480278}},{"probability":0.00858839,"tagId":"b5413c8e-9df4-4d9a-a2c5-dbac5caea08e","tagName":"HardHat","boundingBox":{"left":0.81569105,"top":0.4598646,"width":0.048707247,"height":0.17618498}},{"probability":0.008484437,"tagId":"3bd8b25a-30b9-41d3-901f-e73664f49095","tagName":"SafetyGlass","boundingBox":{"left":0.3138466,"top":0.38765603,"width":0.12779844,"height":0.15008849}},{"probability":0.008350477,"tagId":"6876629c-96a3-4fef-88af-73153acd13a0","tagName":"Person","boundingBox":{"left":0.47818607,"top":0.1040948,"width":0.4635136,"height":0.7415012}},{"probability":0.007910228,"tagId":"67e2ef59-4d5b-4127-9f09-00085ddcb919","tagName":"Shirt","boundingBox":{"left":0.81569105,"top":0.4598646,"width":0.048707247,"height":0.17618498}},{"probability":0.0076070884,"tagId":"6876629c-96a3-4fef-88af-73153acd13a0","tagName":"Person","boundingBox":{"left":0.26766968,"top":0.059939504,"width":0.418383,"height":0.6719135}},{"probability":0.007266716,"tagId":"6876629c-96a3-4fef-88af-73153acd13a0","tagName":"Person","boundingBox":{"left":0.85360104,"top":0.45138374,"width":0.11804825,"height":0.43114552}},{"probability":0.007260954,"tagId":"d6720b77-2150-40f3-8707-d70b766a2911","tagName":"Vest","boundingBox":{"left":0.85943055,"top":0.42174876,"width":0.111328065,"height":0.36172998}},{"probability":0.007110062,"tagId":"b5413c8e-9df4-4d9a-a2c5-dbac5caea08e","tagName":"HardHat","boundingBox":{"left":0.29021895,"top":0.15017729,"width":0.15306118,"height":0.38066232}},{"probability":0.007016064,"tagId":"b5413c8e-9df4-4d9a-a2c5-dbac5caea08e","tagName":"HardHat","boundingBox":{"left":0.2845759,"top":0.0,"width":0.37941602,"height":0.097849384}},{"probability":0.006605696,"tagId":"3bd8b25a-30b9-41d3-901f-e73664f49095","tagName":"SafetyGlass","boundingBox":{"left":0.44934887,"top":0.19859204,"width":0.16465926,"height":0.52737725}},{"probability":0.0065144766,"tagId":"3bd8b25a-30b9-41d3-901f-e73664f49095","tagName":"SafetyGlass","boundingBox":{"left":0.9603284,"top":0.5527964,"width":0.039670587,"height":0.24194717}},{"probability":0.006485251,"tagId":"67e2ef59-4d5b-4127-9f09-00085ddcb919","tagName":"Shirt","boundingBox":{"left":0.85062253,"top":0.46367824,"width":0.052606285,"height":0.18794453}},{"probability":0.005973774,"tagId":"b5413c8e-9df4-4d9a-a2c5-dbac5caea08e","tagName":"HardHat","boundingBox":{"left":0.8745537,"top":0.51273024,"width":0.09484196,"height":0.16288257}},{"probability":0.005928838,"tagId":"6876629c-96a3-4fef-88af-73153acd13a0","tagName":"Person","boundingBox":{"left":0.4102777,"top":0.7318763,"width":0.44337574,"height":0.18766779}},{"probability":0.005894164,"tagId":"67e2ef59-4d5b-4127-9f09-00085ddcb919","tagName":"Shirt","boundingBox":{"left":0.11138326,"top":0.0,"width":0.42763615,"height":0.7077935}},{"probability":0.005755078,"tagId":"b5413c8e-9df4-4d9a-a2c5-dbac5caea08e","tagName":"HardHat","boundingBox":{"left":0.9603284,"top":0.5527964,"width":0.039670587,"height":0.24194717}},{"probability":0.005555544,"tagId":"67e2ef59-4d5b-4127-9f09-00085ddcb919","tagName":"Shirt","boundingBox":{"left":0.44688302,"top":0.036794603,"width":0.53306615,"height":0.7520489}},{"probability":0.005547128,"tagId":"d6720b77-2150-40f3-8707-d70b766a2911","tagName":"Vest","boundingBox":{"left":0.9603284,"top":0.5527964,"width":0.039670587,"height":0.24194717}},{"probability":0.0053816983,"tagId":"6249d7a9-4c08-4ab5-abfb-3ea620e3cd84","tagName":"ForkLift","boundingBox":{"left":0.8559621,"top":0.0,"width":0.14403689,"height":0.6834922}},{"probability":0.0053420523,"tagId":"6249d7a9-4c08-4ab5-abfb-3ea620e3cd84","tagName":"ForkLift","boundingBox":{"left":0.50600564,"top":0.0,"width":0.49399334,"height":0.8454651}},{"probability":0.005191083,"tagId":"3bd8b25a-30b9-41d3-901f-e73664f49095","tagName":"SafetyGlass","boundingBox":{"left":0.4102777,"top":0.7318763,"width":0.44337574,"height":0.18766779}},{"probability":0.0051179635,"tagId":"6876629c-96a3-4fef-88af-73153acd13a0","tagName":"Person","boundingBox":{"left":0.3175477,"top":0.0,"width":0.60785294,"height":0.5048139}}]}
```