# Spark Classfication model to predict titanic survirors

## Spark code to predict classification for practical dataset - titanic survivor dataset

## Use Case

- Use Titanic dataset
- Dataset is available in this folder called Titanic.csv
- Upload the data set to Azure storage

## Prerequistie

- Azure account
- Azure storage account
- Create a container called titanic and upload the Titanic.csv file to container
- Create Azure Data bricks resource
- Create a azure key vault
- Store the primary key into a secret
- Configure the keyvault to azure databricks

## Code

- First lets include the libraries

```
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import functions as f
```

- Load the secrets from keyvault

```
storagekey = dbutils.secrets.get(scope = "allsecrects", key = "storagekey")
```

- Now configure storage account

```
spark.conf.set(
  "fs.azure.account.key.storageaccountname.blob.core.windows.net",
  storagekey)
```

- Now load the data

```
titanicds = spark.read.option("header","true").option("inferSchema", "true").csv("wasbs://titanic@storageacctname.blob.core.windows.net/Titanic.csv")
```

- Display the data set

```
display(titanicds)
```

- Display the schema for dataset

```
display(titanicds.printSchema())
```

- only 5 columns should be string, remaining all columns should be numeric

- Fill 0 for null

```
titanicds1 = titanicds.na.fill(0)
```

- Convert all categorical columns to string index

```
categoricalColumns = [item[0] for item in titanicds1.dtypes if item[1].startswith('string') ]
```

- Create a columns names with all other numerical columns

```
featurescol = ["PassengerId", "Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
```

- Create the stages
- we will combine the string indexers and numerical columns to form combined features

```
stages = []
#iterate through all categorical values
for categoricalCol in categoricalColumns:
    #create a string indexer for those categorical values and assign a new name including the word 'Index'
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')

    #append the string Indexer to our list of stages
    stages += [stringIndexer]
```

- now create the label indexer
- Create the vector assembler for numeric data.
- Add all those to stages
- Train the model
- predict the model with test data
- print the accuracy

```
labelIndexer = StringIndexer(inputCol="Survived", outputCol="indexedLabel").fit(titanicds1)

#assembler = VectorAssembler(inputCols=featurescol, outputCol="features")
assembler = VectorAssembler(inputCols=featurescol, outputCol="features")

#featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(titanicds1)


(trainingData, testData) = titanicds1.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)


# Chain indexers and forest in a Pipeline
#pipeline = Pipeline(stages=[labelIndexer, assembler, rf, labelConverter])
stages += [labelIndexer]
stages += [assembler]
stages += [rf]
stages += [labelConverter]

pipeline = Pipeline(stages=stages)

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "Survived", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = ", accuracy)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only
```