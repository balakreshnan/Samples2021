# Process Form Recognizer to Text

## Using Azure Cognitive Services Forms and Logic apps

## No Code - Workflow style

## Pre requistie

- Azure Account
- Azure Storage account
- Azure Cognitive Services - Form Recognizer
- Azure Logic apps
- Get connection string for storage
- Get the primary key to be used as subcription key for cognitive services
- Audio file should be pdf format
- Audio file cannot be too big

## Full Flow

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/form11.jpg "Service Health")

## Logic apps

- First create a trigger from Blob

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/form1.jpg "Service Health")

- input container name: pdfinput
- Read Blob contents
- Container name: pdfinput
- blob name - specifiy the blob name

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/form2.jpg "Service Health")

- Bring HTTP action
- method: POST
- URI - https://cogsvcname.cognitiveservices.azure.com/formrecognizer/v2.1/layout/analyze
- Headers needed
- sub - Ocp-Apim-Subscription-Key
- Select Blob response data from previous actions

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/form3.jpg "Service Health")

- Delete the existing header info storage
- Specifiy Container name and blob name
- Container name - podoutput
- blob name - pdfoutputanalyze.json

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/form4.jpg "Service Health")

- Now upload the new header contents to blob
- Container name
- Blob name
- Contents: Headers (Select)

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/form5.jpg "Service Health")

- Now introduce a delay for API to process
- This can be customized to how big the file are
- Select unit as Selcong
- For count = 45 - 45 seconds

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/form6.jpg "Service Health")

- Now bring HTTP action to retrive results
- details
- URL formation

```
concat('https://cushwakepdf.cognitiveservices.azure.com/formrecognizer/v2.1/layout/analyzeResults/', outputs('HTTP')?['headers']?['apim-request-id'])
```

- Add headers as Ocp-Apim-Subscription-Key - This is the primary/secondary key

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/form7.jpg "Service Health")

- Now parse JSON
- Bring parseJSON actions
- Select the above output as content
- Content: Body
- for Schema

```
{
    "properties": {
        "analyzeResult": {
            "properties": {
                "pageResults": {
                    "items": {
                        "properties": {
                            "page": {
                                "type": "integer"
                            },
                            "tables": {
                                "items": {
                                    "properties": {
                                        "boundingBox": {
                                            "items": {
                                                "type": "number"
                                            },
                                            "type": "array"
                                        },
                                        "cells": {
                                            "items": {
                                                "properties": {
                                                    "boundingBox": {
                                                        "items": {
                                                            "type": "number"
                                                        },
                                                        "type": "array"
                                                    },
                                                    "columnIndex": {
                                                        "type": "integer"
                                                    },
                                                    "elements": {
                                                        "type": "array"
                                                    },
                                                    "isHeader": {
                                                        "type": "boolean"
                                                    },
                                                    "rowIndex": {
                                                        "type": "integer"
                                                    },
                                                    "text": {
                                                        "type": "string"
                                                    }
                                                },
                                                "required": [
                                                    "rowIndex",
                                                    "columnIndex",
                                                    "text",
                                                    "boundingBox",
                                                    "elements",
                                                    "isHeader"
                                                ],
                                                "type": "object"
                                            },
                                            "type": "array"
                                        },
                                        "columns": {
                                            "type": "integer"
                                        },
                                        "rows": {
                                            "type": "integer"
                                        }
                                    },
                                    "required": [
                                        "rows",
                                        "columns",
                                        "cells",
                                        "boundingBox"
                                    ],
                                    "type": "object"
                                },
                                "type": "array"
                            }
                        },
                        "required": [
                            "page",
                            "tables"
                        ],
                        "type": "object"
                    },
                    "type": "array"
                },
                "readResults": {
                    "items": {
                        "properties": {
                            "angle": {
                                "type": "number"
                            },
                            "height": {
                                "type": "number"
                            },
                            "lines": {
                                "items": {
                                    "properties": {
                                        "appearance": {
                                            "properties": {
                                                "style": {
                                                    "properties": {
                                                        "confidence": {
                                                            "type": "number"
                                                        },
                                                        "name": {
                                                            "type": "string"
                                                        }
                                                    },
                                                    "type": "object"
                                                }
                                            },
                                            "type": "object"
                                        },
                                        "boundingBox": {
                                            "items": {
                                                "type": "number"
                                            },
                                            "type": "array"
                                        },
                                        "text": {
                                            "type": "string"
                                        },
                                        "words": {
                                            "items": {
                                                "properties": {
                                                    "boundingBox": {
                                                        "items": {
                                                            "type": "number"
                                                        },
                                                        "type": "array"
                                                    },
                                                    "confidence": {
                                                        "type": "number"
                                                    },
                                                    "text": {
                                                        "type": "string"
                                                    }
                                                },
                                                "required": [
                                                    "boundingBox",
                                                    "text",
                                                    "confidence"
                                                ],
                                                "type": "object"
                                            },
                                            "type": "array"
                                        }
                                    },
                                    "required": [
                                        "boundingBox",
                                        "text",
                                        "appearance",
                                        "words"
                                    ],
                                    "type": "object"
                                },
                                "type": "array"
                            },
                            "page": {
                                "type": "integer"
                            },
                            "selectionMarks": {
                                "items": {
                                    "properties": {
                                        "boundingBox": {
                                            "items": {
                                                "type": "number"
                                            },
                                            "type": "array"
                                        },
                                        "confidence": {
                                            "type": "number"
                                        },
                                        "state": {
                                            "type": "string"
                                        }
                                    },
                                    "required": [
                                        "boundingBox",
                                        "confidence",
                                        "state"
                                    ],
                                    "type": "object"
                                },
                                "type": "array"
                            },
                            "unit": {
                                "type": "string"
                            },
                            "width": {
                                "type": "number"
                            }
                        },
                        "required": [
                            "page",
                            "angle",
                            "width",
                            "height",
                            "unit",
                            "lines",
                            "selectionMarks"
                        ],
                        "type": "object"
                    },
                    "type": "array"
                },
                "version": {
                    "type": "string"
                }
            },
            "type": "object"
        },
        "createdDateTime": {
            "type": "string"
        },
        "lastUpdatedDateTime": {
            "type": "string"
        },
        "status": {
            "type": "string"
        }
    },
    "type": "object"
}
```

- Delete the output blob

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/form8.jpg "Service Health")

- Now delete the old data
- I am doing since its the same file name and connector doesn't allow overwrite
- Container Name: podoutput
- Blob Name: formoutput.json

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/form9.jpg "Service Health")

- Write to output
- Container Name: podoutput
- Blob Name: formoutput.json
- Content: Select Body from parseJSON

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/form10.jpg "Service Health")