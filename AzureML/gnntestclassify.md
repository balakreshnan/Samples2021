# Graph Neural network in Azure Machine learning

## How to run graph neural network in Azure machine learning (Classification)

## Prerequistie

- Azure account
- Azure Machine learning service
- Create a compute instance
- Need a storage account

## Use Case

- Create Deep learning models using graph
- Technology is brand new and subject to change
- Idea here is to use graph data and use deep learning
- using spektral (https://github.com/danielegrattarola/spektral)

## Example 1

- There are more example in the above web site
- Goal here is to run spektral in Azure machine learning service
- Create a compute instance

## Code

- Now install the library

```
pip install spektral
```

- installed version at the time of the tutorial

```
pip show spektral
```

```
Name: spektral
Version: 1.0.8
Summary: Graph Neural Networks with Keras and Tensorflow 2.
Home-page: https://github.com/danielegrattarola/spektral
Author: Daniele Grattarola
Author-email: daniele.grattarola@gmail.com
License: MIT
Location: /anaconda/envs/azureml_py36/lib/python3.6/site-packages
Requires: networkx, scipy, tensorflow, numpy, lxml, scikit-learn, requests, joblib, pandas, tqdm
Required-by: 
Note: you may need to restart the kernel to use updated packages.
```

- Now create a new jupyter notebook with python 3.6 and AML sdk
- Import library

```
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader
from spektral.datasets import TUDataset
from spektral.models import GeneralGNN
```

- Set the CPU/GPU and variables

```
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

################################################################################
# Config
################################################################################
batch_size = 32
learning_rate = 0.01
epochs = 400
```

- Now download sample data

```
data = TUDataset("PROTEINS")

# Train/test split
np.random.shuffle(data)
split = int(0.8 * len(data))
data_tr, data_te = data[:split], data[split:]
```

- Split the data

```
# Data loaders
loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_te = DisjointLoader(data_te, batch_size=batch_size)
```

- define the model

```
model = GeneralGNN(data.n_labels, activation="softmax")
optimizer = Adam(learning_rate)
loss_fn = CategoricalCrossentropy()
```

- Create the model fit functions

```
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(categorical_accuracy(target, predictions))
    return loss, acc


def evaluate(loader):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(categorical_accuracy(target, pred)),
            len(target),  # Keep track of batch size
        )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])
```

- now train the model
- Model will run 400 epochs

```
epoch = step = 0
results = []
for batch in loader_tr:
    step += 1
    loss, acc = train_step(*batch)
    results.append((loss, acc))
    if step == loader_tr.steps_per_epoch:
        step = 0
        epoch += 1
        results_te = evaluate(loader_te)
        print(
            "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Test loss: {:.3f} - Test acc: {:.3f}".format(
                epoch, *np.mean(results, 0), *results_te
            )
        )
        results = []
```

- Now show the evaluted results from the model above

```
results_te = evaluate(loader_te)
print("Final results - Loss: {:.3f} - Acc: {:.3f}".format(*results_te))
```