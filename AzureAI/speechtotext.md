# Process Speech and convert to Text

## Using Azure Cognitive Services Speech to Text and Logic apps

## No Code - Workflow style

## Pre requistie

- Azure Account
- Azure Storage account
- Azure Cognitive Services
- Azure Logic apps
- Get connection string for storage
- Get the primary key to be used as subcription key for cognitive services
- Audio file should be wav format
- Audio file cannot be too big
- Audio time 10 min

## Logic apps

- First create a trigger from Blob

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext2.jpg "Service Health")

- Create a connection string using blob connection string

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext3.jpg "Service Health")

- Now bring "Reads Blob Content from Azure Storage"

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext4.jpg "Service Health")

- Container name: audioinput
- Choose dynamics and select the blob name as above picture
- Bring HTTP action
- Here we need to call the speech to text and pass the parameter
- Accept: application/json;text/xml
- Content-type: audio/wav; codecs=audio/pcm; samplerate=16000
- Expect: 100-continue
- Ocp-Apim-Subscription-Key: xxxx-xxxxxx-xxxxxx-xxxx
- Transfer-Encoding: chunked

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext5.jpg "Service Health")

- For Body Choose the read blob content
- This should pass the audio binary content to cognitive service api
- Now lets parseJSON the api output
- Now select the body from http output
- Provide the schema as

```
{
    "properties": {
        "Duration": {
            "type": "integer"
        },
        "NBest": {
            "items": {
                "properties": {
                    "Confidence": {
                        "type": "number"
                    },
                    "Display": {
                        "type": "string"
                    },
                    "ITN": {
                        "type": "string"
                    },
                    "Lexical": {
                        "type": "string"
                    },
                    "MaskedITN": {
                        "type": "string"
                    }
                },
                "required": [
                    "Confidence",
                    "Lexical",
                    "ITN",
                    "MaskedITN",
                    "Display"
                ],
                "type": "object"
            },
            "type": "array"
        },
        "Offset": {
            "type": "integer"
        },
        "RecognitionStatus": {
            "type": "string"
        }
    },
    "type": "object"
}
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext6.jpg "Service Health")

- Now add action for upload data to blob
- Give a container output
- Give a output name

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext7.jpg "Service Health")

- Go to Overview and then click Run Trigger and click -> Run
- Upload the wav file
- Wait for it to process

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext1.jpg "Service Health")

- Give some time for the Speech API to process