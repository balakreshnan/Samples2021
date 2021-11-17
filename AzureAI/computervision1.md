# Azure Cognitive Services Computer Vision and Custom Vision

## Use Computer vision to identify objects in an image

## Use Case

- Indentify objects in an image which are common in a range of scenarios
- Check for condition on objects detected For Example if certain objects are present in an image
- Use Custom vision to detect more custom objects
- use other congitive services to infuse AI into the process
- Event based

## Architecture

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision21.jpg "Service Health")

## Logic Flow

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision22.jpg "Service Health")
![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision23.jpg "Service Health")
![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision24.jpg "Service Health")

## Code

- Configure Azure Storage

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision6.jpg "Service Health")

- Now sent to Analyze image

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision8.jpg "Service Health")

- Delete existing files

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision25.jpg "Service Health")

- Save the output

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision26.jpg "Service Health")

- Parse JSON

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision27.jpg "Service Health")


```
{
    "properties": {
        "categories": {
            "items": {
                "properties": {
                    "detail": {
                        "properties": {
                            "celebrities": {
                                "type": "array"
                            }
                        },
                        "type": "object"
                    },
                    "name": {
                        "type": "string"
                    },
                    "score": {
                        "type": "number"
                    }
                },
                "required": [
                    "name",
                    "score",
                    "detail"
                ],
                "type": "object"
            },
            "type": "array"
        },
        "description": {
            "properties": {
                "captions": {
                    "items": {
                        "properties": {
                            "confidence": {
                                "type": "number"
                            },
                            "text": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "text",
                            "confidence"
                        ],
                        "type": "object"
                    },
                    "type": "array"
                },
                "tags": {
                    "items": {
                        "type": "string"
                    },
                    "type": "array"
                }
            },
            "type": "object"
        },
        "faces": {
            "items": {
                "properties": {
                    "age": {
                        "type": "integer"
                    },
                    "faceRectangle": {
                        "properties": {
                            "height": {
                                "type": "integer"
                            },
                            "left": {
                                "type": "integer"
                            },
                            "top": {
                                "type": "integer"
                            },
                            "width": {
                                "type": "integer"
                            }
                        },
                        "type": "object"
                    },
                    "gender": {
                        "type": "string"
                    }
                },
                "required": [
                    "age",
                    "gender",
                    "faceRectangle"
                ],
                "type": "object"
            },
            "type": "array"
        },
        "metadata": {
            "properties": {
                "format": {
                    "type": "string"
                },
                "height": {
                    "type": "integer"
                },
                "width": {
                    "type": "integer"
                }
            },
            "type": "object"
        },
        "modelVersion": {
            "type": "string"
        },
        "objects": {
            "items": {
                "properties": {
                    "confidence": {
                        "type": "number"
                    },
                    "object": {
                        "type": "string"
                    },
                    "parent": {
                        "properties": {
                            "confidence": {
                                "type": "number"
                            },
                            "object": {
                                "type": "string"
                            }
                        },
                        "type": "object"
                    },
                    "rectangle": {
                        "properties": {
                            "h": {
                                "type": "integer"
                            },
                            "w": {
                                "type": "integer"
                            },
                            "x": {
                                "type": "integer"
                            },
                            "y": {
                                "type": "integer"
                            }
                        },
                        "type": "object"
                    }
                },
                "required": [
                    "rectangle",
                    "object",
                    "confidence"
                ],
                "type": "object"
            },
            "type": "array"
        },
        "requestId": {
            "type": "string"
        },
        "tags": {
            "items": {
                "properties": {
                    "confidence": {
                        "type": "number"
                    },
                    "name": {
                        "type": "string"
                    }
                },
                "required": [
                    "name",
                    "confidence"
                ],
                "type": "object"
            },
            "type": "array"
        }
    },
    "type": "object"
}
```

- Set the condition

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision28.jpg "Service Health")

- If True then send to custom vision
- Send to custom vision

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision29.jpg "Service Health")

- Delete existing files

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision30.jpg "Service Health")

- Save the data to blob as json

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/computervision31.jpg "Service Health")