# Azure Machine learning Service Designer - Data Engineering

## How can we do Data engineering in Azure Machine Learning Service using designer

## Prerequistie

- Azure Account
- Azure Storage
- Azure Machine learning Service

## Introduction

- This tutorial is only to show how to do data engineering in Azure Machine Learning Service using designer.
- Data used is Titanic dataset. which is a famous dataset in Machine Learning.
- Open source dataset is used here.
- Every task or flow item has parameters and output
- After run every task output can be visualized
- Output will change based on the task or flow item

## Overall flow

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer1.jpg "Service Health")

- Above is the overall experiment
- Build using low code environment
- All are drag and drop

## What's done

### Bring the dataset

### Select columns in dataset

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer2.jpg "Service Health")

### Execute python script - Correlation Chart

```
    import seaborn as sn
    import matplotlib.pyplot as plt

    corrMatrix = dataframe1.corr()
    print (corrMatrix)
    sn.heatmap(corrMatrix, annot=True)
    plt.show()
    img_file = "corrchart1.png"
    plt.savefig(img_file)

    from azureml.core import Run
    run = Run.get_context(allow_offline=True)
    run.upload_file(f"graphics/{img_file}", img_file)
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer3.jpg "Service Health")

- Output

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer4.jpg "Service Health")

### Execute python script - Covariance Chart

```
    covMatrix = dataframe1.cov()
    print (covMatrix)
    sn.heatmap(covMatrix, annot=True)
    plt.show()
    img_file = "covchart1.png"
    plt.savefig(img_file)

    from azureml.core import Run
    run = Run.get_context(allow_offline=True)
    run.upload_file(f"graphics/{img_file}", img_file)
```

- Code

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer5.jpg "Service Health")

- Output

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer6.jpg "Service Health")

### Remove duplicate rows

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer7.jpg "Service Health")

### Normalize data

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer8.jpg "Service Health")

### Group data in bins

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer9.jpg "Service Health")

### Edit Metadata to convert String to Categorical column - Name

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer10.jpg "Service Health")

### Edit Metadata to convert String to Categorical column - Cabin

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer11.jpg "Service Health")

### Edit Metadata to convert String to Categorical column - Embarked

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer12.jpg "Service Health")

### Clip value - Avoid overfitting

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer13.jpg "Service Health")

### Clean missing data

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer14.jpg "Service Health")

### Apply math operations

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer15.jpg "Service Health")

### Split data into training and test data

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer18.jpg "Service Health")

### bring model to train

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer16.jpg "Service Health")

### Train model

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer17.jpg "Service Health")

### Score model

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer19.jpg "Service Health")

- Output

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer20.jpg "Service Health")

### Evaluate Model

- output

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer21.jpg "Service Health")

- Roc Curve

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer22.jpg "Service Health")

- Confusion Matrix

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AMLDesigner/images/designer23.jpg "Service Health")