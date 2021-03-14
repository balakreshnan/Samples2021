# Pytorch Regression model using Azure Machine learning

## Run Regression using pytorch with Azure ML

## Pre requistie

- Azure Account
- Resource group
- Azure Machine learning
- Azure Storage blob
- Download the Nasa Predictive Maintenance data set
- Go to https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
- Scroll down to Turbofan Engine Degradation Simulation Data Set click Download Turbofan Engine Degradation Simulation Data Set (68202 downloads)
- Create a dataset in Azure ML called nasaPredMaint

## Azure Blob Storage

- Create a Storage account
- Create a container called PredMaint
- Upload the files that was downloaded

## Azure Machine Learning

- First create a data set

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/nasaamldataset1.jpg "Service Health")

- Create a new data store called: nasapredmaint

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/nasaamldataset2.jpg "Service Health")

- Now lets create a Compute instance
- Start the compute instance
- Create a new notebook
- Select python + AzureML
- Load the registered data set

```
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = 'xxxxxxxxxxx'
resource_group = 'rgname'
workspace_name = 'mlworkspace'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='nasaPredMaint')
dataset.to_pandas_dataframe()
```

- convert data set to dataframe

```
df = dataset.to_pandas_dataframe()
```

- List all the columns

```
df.columns
```

- import librarbies

```
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
```

- Drop columns which are labels and un wanted columns

```
df1 = df.drop(columns=['timecycles','sensor22', 'sensor23'])
```

- Split features and labels

```
X = df1.iloc[:,:31]
y = df[['timecycles']]
```

- display dataframe

```
df1.head()
```

- now import new libraries

```
#Let's get rid of some imports
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
#Define the model 
import torch
import torch.nn as nn
import torch.nn.functional as F
```

- defines features and label

```
from sklearn.model_selection  import train_test_split
X = df1.iloc[:, 0:29]
y = df[['timecycles']]

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

- Define parameters

```
#Define training hyperprameters.
batch_size = 50
num_epochs = 200
learning_rate = 0.01
size_hidden= 100

#Calculate some other hyperparameters based on data.  
batch_no = len(X_train) // batch_size  #batches
cols=X_train.shape[1] #Number of columns in input matrix
n_output=1
```

- Create a model

```
#Create the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print("Executing the model on :",device)
class Net(torch.nn.Module):
    def __init__(self, n_feature, size_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(cols, size_hidden)   # hidden layer
        self.predict = torch.nn.Linear(size_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x
net = Net(cols, size_hidden, n_output)
```

- configure the optimizer

```
#Adam is a specific flavor of gradient decent which is typically better
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
criterion = torch.nn.MSELoss(size_average=False)  # this is for regression mean squared loss
```

- change to values

```
#Change to numpy arraay. 
X_train=X_train.values
y_train=y_train.values
X_test=X_test.values
y_test=y_test.values
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/nasaamldataset3.jpg "Service Health")

- now run the model

```
from sklearn.utils import shuffle
from torch.autograd import Variable
running_loss = 0.0
for epoch in range(num_epochs):
    #Shuffle just mixes up the dataset between epocs
    X_train, y_train = shuffle(X_train, y_train)
    # Mini batch learning
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        inputs = Variable(torch.FloatTensor(X_train[start:end]))
        labels = Variable(torch.FloatTensor(y_train[start:end]))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print("outputs",outputs)
        #print("outputs",outputs,outputs.shape,"labels",labels, labels.shape)
        loss = criterion(outputs, torch.unsqueeze(labels,dim=1))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
    print('Epoch {}'.format(epoch+1), "loss: ",running_loss)
    running_loss = 0.0
```

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/Images/nasaamldataset4.jpg "Service Health")

- Calculate the metrics

```
import pandas as pd
from sklearn.metrics import r2_score

X = Variable(torch.FloatTensor(X_train)) 
result = net(X)
pred=result.data[:,0].numpy()
print(len(pred),len(y_train))
r2_score(pred,y_train)
```

- Find R2 score

```
import pandas as pd
from sklearn.metrics import r2_score
#This is a little bit tricky to get the resulting prediction.  
def calculate_r2(x,y=[]):
    """
    This function will return the r2 if passed x and y or return predictions if just passed x. 
    """
    # Evaluate the model with the test set. 
    X = Variable(torch.FloatTensor(x))  
    result = net(X) #This outputs the value for regression
    result=result.data[:,0].numpy()
  
    if len(y) != 0:
        r2=r2_score(result, y)
        print("R-Squared", r2)
        #print('Accuracy {:.2f}'.format(num_right / len(y)), "for a total of ", len(y), "records")
        return pd.DataFrame(data= {'actual': y, 'predicted': result})
    else:
        print("returning predictions")
        return result
```

- Run linear model

```
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit( X_train, y_train )
```

- Print score

```
print('R2 for Train)', lm.score( X_train, y_train ))
print('R2 for Test (cross validation)', lm.score(X_test, y_test))
```