# Azure AI document translation

## Translate document in batch with Cognitive services

- Batch translate pdf document from one language to another

## Use Case

- The below architecture shows how to do batch document translations
- Here i have translationinput as input container for document
- Then i created translationoutput as output container to store the output
- For both create SAS key with Read and write.
- Atleast read for translationinput
- Write for translationoutput

## Architecture

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/DocumentTranslation/images/doctrans1.jpg "Service Health")

## Steps

- Create a new logic app
- Create a new workload
- create a new connection for blob storage where you have the data
- to trigger the flow start when a file is uploaded
- i am reading the pdf file - this step is not necessary

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/DocumentTranslation/images/doctrans2.jpg "Service Health")

- Now drag HTTP to send the translation request
- Provide Ocp-Apim-Subscription-Key key which is from cognitive services
- then provide the URL

```
https://cognitivesvcname.cognitiveservices.azure.com/translator/text/batch/v1.0/batches
```

- Next configure the input and outputs
- Here i am converting to 2 different languages

```
{
  "inputs": [
    {
      "source": {
        "sourceUrl": "https://storagename.blob.core.windows.net/translateinput?sp=racwl&st=2020-10-19T21:27:53Z&se=2020-10-21T05:27:53Z&spr=https&sv=2020-08-04&sr=c&sig=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
      },
      "targets": [
        {
          "language": "fr",
          "targetUrl": "https://storagename.blob.core.windows.net/translateoutput?sp=racwl&st=2020-10-19T21:29:02Z&se=2020-10-21T05:29:02Z&spr=https&sv=2020-08-04&sr=c&sig=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        },
        {
          "language": "id",
          "targetUrl": "https://storaegname.blob.core.windows.net/translateoutput?sp=racwl&st=2020-10-19T21:29:02Z&se=2020-10-21T05:29:02Z&spr=https&sv=2020-08-04&sr=c&sig=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        }
      ]
    }
  ]
}
```

- source and targets are full blob sas keys
- the above SAS keys are fake and just for sample

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/DocumentTranslation/images/doctrans3.jpg "Service Health")

- Next wait for 30 seconds

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/DocumentTranslation/images/doctrans4.jpg "Service Health")

- Now lets get the status
- From the above batch submission, response header will have the URL to query for status

```
@{outputs('HTTP')['headers']?['Operation-Location']}
```

- Make sure send the Ocp-Apim-Subscription-Key

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/DocumentTranslation/images/doctrans5.jpg "Service Health")

- Now save and run the logic app.
- Once successful then go out storage explorer and see the translationoutput container for processed pdf file.
- Done