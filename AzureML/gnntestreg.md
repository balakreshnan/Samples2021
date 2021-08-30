# Graph Neural network in Azure Machine learning

## How to run graph neural network in Azure machine learning (Regression)

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
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader
from spektral.datasets import QM9
from spektral.layers import ECCConv, GlobalSumPool
```

- Set the CPU/GPU and variables

```
learning_rate = 1e-3  # Learning rate
epochs = 10  # Number of training epochs
batch_size = 32  # Batch size
```

- Now download sample data

```
dataset = QM9(amount=1000)  # Set amount=None to train on whole dataset

# Parameters
F = dataset.n_node_features  # Dimension of node features
S = dataset.n_edge_features  # Dimension of edge features
n_out = dataset.n_labels  # Dimension of the target
```

- Split the data

```
idxs = np.random.permutation(len(dataset))
split = int(0.9 * len(dataset))
idx_tr, idx_te = np.split(idxs, [split])
dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]

loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
loader_te = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1)
```

- define the model

```
class Net(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = ECCConv(32, activation="relu")
        self.conv2 = ECCConv(32, activation="relu")
        self.global_pool = GlobalSumPool()
        self.dense = Dense(n_out)

    def call(self, inputs):
        x, a, e, i = inputs
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        output = self.global_pool([x, i])
        output = self.dense(output)

        return output


model = Net()
optimizer = Adam(learning_rate)
loss_fn = MeanSquaredError()
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
    return loss
```

- now train the model
- Model will run 400 epochs

```
step = loss = 0
for batch in loader_tr:
    step += 1
    loss += train_step(*batch)
    if step == loader_tr.steps_per_epoch:
        step = 0
        print("Loss: {}".format(loss / loader_tr.steps_per_epoch))
        loss = 0
```

- Now show the evaluted results from the model above

```
print("Testing model")
loss = 0
for batch in loader_te:
    inputs, target = batch
    predictions = model(inputs, training=False)
    loss += loss_fn(target, predictions)
loss /= loader_te.steps_per_epoch
print("Done. Test loss: {}".format(loss))
```