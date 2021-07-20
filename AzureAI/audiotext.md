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
- Now go to blob storage

```
{
  "RecognitionStatus": "Success",
  "Offset": 300000,
  "Duration": 524000000,
  "NBest": [
    {
      "Confidence": 0.972784698009491,
      "Lexical": "the speech SDK exposes many features from the speech service but not all of them the capabilities of the speech SDK are often associated with scenarios the speech SDK is ideal for both real time and non real time scenarios using local devices files azure blob storage and even input and output streams when a scenario is not achievable with the speech SDK look for a rest API alternative speech to text also known as speech recognition transcribes audio streams to text that your applications tools or devices can consume more display use speech to text with language understanding louis to deride user intents from transcribed speech and act on voice commands you speech translation to translate speech input to a different language with a single call for more information see speech to text basics",
      "ITN": "the speech SDK exposes many features from the speech service but not all of them the capabilities of the speech SDK are often associated with scenarios the speech SDK is ideal for both real time and non real time scenarios using local devices files azure blob storage and even input and output streams when a scenario is not achievable with the speech SDK look for a rest API alternative speech to text also known as speech recognition transcribes audio streams to text that your applications tools or devices can consume more display use speech to text with language understanding louis to deride user intents from transcribed speech and act on voice commands you speech translation to translate speech input to a different language with a single call for more information see speech to text basics",
      "MaskedITN": "the speech sdk exposes many features from the speech service but not all of them the capabilities of the speech sdk are often associated with scenarios the speech sdk is ideal for both real time and non real time scenarios using local devices files azure blob storage and even input and output streams when a scenario is not achievable with the speech sdk look for a rest api alternative speech to text also known as speech recognition transcribes audio streams to text that your applications tools or devices can consume more display use speech to text with language understanding louis to deride user intents from transcribed speech and act on voice commands you speech translation to translate speech input to a different language with a single call for more information see speech to text basics",
      "Display": "The Speech SDK exposes many features from the speech service, but not all of them. The capabilities of the speech SDK are often associated with scenarios. The Speech SDK is ideal for both real time and non real time scenarios using local devices files, Azure blob storage and even input and output streams. When a scenario is not achievable with the speech SDK, look for a rest API. Alternative speech to text, also known as speech recognition, transcribes audio streams to text that your applications, tools or devices can consume more display use speech to text with language, understanding Louis to deride user intents from transcribed speech and act on voice commands. You speech translation to translate speech input to a different language with a single call. For more information, see speech to text basics."
    }
  ]
}
```

- Above is the sample output
- Confidence score and Display is available
- Now process Text analytics and pull Keypharses, PII, Sentiment and Entities
- Create 3 variable one for id, text and language
- Create id

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext8.jpg "Service Health")

- Create language

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext9.jpg "Service Health")

- Create Text

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext10.jpg "Service Health")

- Next is compose

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext11.jpg "Service Health")

```
{
  "documents": [
    {
      "id": @{variables('id')},
      "language": @{variables('language')},
      "text": @{variables('text')}
    }
  ]
}
```

- text analytics API

```
https://cogsvcname.cognitiveservices.azure.com/text/analytics/v3.1/keyPhrases
```

- Provide Header - Ocp-Apim-Subscription-Key
- Headers - Content-Type
- Body - Content from compose output

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext12.jpg "Service Health")

- Parse JSON output

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext13.jpg "Service Health")

- Schema 

```
{
    "properties": {
        "documents": {
            "items": {
                "properties": {
                    "id": {
                        "type": "string"
                    },
                    "keyPhrases": {
                        "items": {
                            "type": "string"
                        },
                        "type": "array"
                    },
                    "warnings": {
                        "type": "array"
                    }
                },
                "required": [
                    "id",
                    "keyPhrases",
                    "warnings"
                ],
                "type": "object"
            },
            "type": "array"
        },
        "errors": {
            "type": "array"
        },
        "modelVersion": {
            "type": "string"
        }
    },
    "type": "object"
}
```

- Delete the blob
- name of blob: textanalytics.json

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext14.jpg "Service Health")

- Save the blob now
- name of blob: textanalytics.json

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext15.jpg "Service Health")

- Now call Text analytics for PII

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext16.jpg "Service Health")

```
https://cogsvcnmae.cognitiveservices.azure.com/text/analytics/v3.1/entities/recognition/pii
```

- Provide Header - Ocp-Apim-Subscription-Key
- Headers - Content-Type
- Body - Content from compose output

- Now bring parseJSON

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext25.jpg "Service Health")

```
{
    "type": "object",
    "properties": {
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "redactedText": {
                        "type": "string"
                    },
                    "id": {
                        "type": "string"
                    },
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string"
                                },
                                "category": {
                                    "type": "string"
                                },
                                "offset": {
                                    "type": "integer"
                                },
                                "length": {
                                    "type": "integer"
                                },
                                "confidenceScore": {
                                    "type": "number"
                                }
                            },
                            "required": [
                                "text",
                                "category",
                                "offset",
                                "length",
                                "confidenceScore"
                            ]
                        }
                    },
                    "warnings": {
                        "type": "array"
                    }
                },
                "required": [
                    "redactedText",
                    "id",
                    "entities",
                    "warnings"
                ]
            }
        },
        "errors": {
            "type": "array"
        },
        "modelVersion": {
            "type": "string"
        }
    }
}
```

- now bring delete
- blob name: textpii.json

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext17.jpg "Service Health")

- now save the file to blob
- blob name: textpii.json

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext18.jpg "Service Health")

- now get Sentiment API

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext19.jpg "Service Health")

```
https://cogsvcnmae.cognitiveservices.azure.com/text/analytics/v3.1/sentiment
```

- Provide Header - Ocp-Apim-Subscription-Key
- Headers - Content-Type
- Body - Content from compose output

- Bring parseJSON

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext26.jpg "Service Health")

```
{
    "type": "object",
    "properties": {
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string"
                    },
                    "sentiment": {
                        "type": "string"
                    },
                    "confidenceScores": {
                        "type": "object",
                        "properties": {
                            "positive": {
                                "type": "number"
                            },
                            "neutral": {
                                "type": "number"
                            },
                            "negative": {
                                "type": "number"
                            }
                        }
                    },
                    "sentences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sentiment": {
                                    "type": "string"
                                },
                                "confidenceScores": {
                                    "type": "object",
                                    "properties": {
                                        "positive": {
                                            "type": "number"
                                        },
                                        "neutral": {
                                            "type": "number"
                                        },
                                        "negative": {
                                            "type": "number"
                                        }
                                    }
                                },
                                "offset": {
                                    "type": "integer"
                                },
                                "length": {
                                    "type": "integer"
                                },
                                "text": {
                                    "type": "string"
                                }
                            },
                            "required": [
                                "sentiment",
                                "confidenceScores",
                                "offset",
                                "length",
                                "text"
                            ]
                        }
                    },
                    "warnings": {
                        "type": "array"
                    }
                },
                "required": [
                    "id",
                    "sentiment",
                    "confidenceScores",
                    "sentences",
                    "warnings"
                ]
            }
        },
        "errors": {
            "type": "array"
        },
        "modelVersion": {
            "type": "string"
        }
    }
}
```

- now bring delete
- blob name: textsentiment.json

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext20.jpg "Service Health")

- now bring save blob
- blob name: textsentiment.json

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext21.jpg "Service Health")

- now get the entities

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext22.jpg "Service Health")

```
https://cogsvcnmae.cognitiveservices.azure.com/text/analytics/v3.1/entities/recognition/general
```

- Provide Header - Ocp-Apim-Subscription-Key
- Headers - Content-Type
- Body - Content from compose output

- Bring parseJSON

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext27.jpg "Service Health")

```
{
    "type": "object",
    "properties": {
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string"
                    },
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string"
                                },
                                "category": {
                                    "type": "string"
                                },
                                "subcategory": {
                                    "type": "string"
                                },
                                "offset": {
                                    "type": "integer"
                                },
                                "length": {
                                    "type": "integer"
                                },
                                "confidenceScore": {
                                    "type": "number"
                                }
                            },
                            "required": [
                                "text",
                                "category",
                                "offset",
                                "length",
                                "confidenceScore"
                            ]
                        }
                    },
                    "warnings": {
                        "type": "array"
                    }
                },
                "required": [
                    "id",
                    "entities",
                    "warnings"
                ]
            }
        },
        "errors": {
            "type": "array"
        },
        "modelVersion": {
            "type": "string"
        }
    }
}
```

- now bring delete
- blob name: textentities.json

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext23.jpg "Service Health")

- now save the final output

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/stotext24.jpg "Service Health")