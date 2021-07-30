# Azure ML AutoML Rest API with probablitlies

## Use Custom score script to deploy REST API

## Steps

- Create a automated ML run
- Now navigate to best model
- Go to outputs/logs and select output and download model.pkl
- Download the score file also
- Edit the score file 

```
# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

data_sample = PandasParameterType(pd.DataFrame({"PassengerId": pd.Series([0], dtype="int64"), "Pclass": pd.Series([0], dtype="int64"), "Name": pd.Series(["example_value"], dtype="object"), "Sex": pd.Series(["example_value"], dtype="object"), "Age": pd.Series([0.0], dtype="float64"), "SibSp": pd.Series([0], dtype="int64"), "Parch": pd.Series([0], dtype="int64"), "Ticket": pd.Series(["example_value"], dtype="object"), "Fare": pd.Series([0.0], dtype="float64"), "Cabin": pd.Series(["example_value"], dtype="object"), "Embarked": pd.Series(["example_value"], dtype="object")}))
input_sample = StandardPythonParameterType({'data': data_sample})

result_sample = NumpyParameterType(np.array([0]))
output_sample = StandardPythonParameterType({'Results':result_sample})

try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script_v2')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")     
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('Inputs', input_sample)
@output_schema(output_sample)
def run(Inputs):
    data = Inputs['data']
    result = model.predict_proba(data)
    return result.tolist()
```

- Now if you want to manual test in jupyter notebook
- upload the model.pkl file to jupyterlab folder
- Create a sample titanic4.json file and add these content below

```
[{"PassengerId":"1","Pclass":"3","Name":"Braund Mr. Owen Harris","Sex":"male","Age":"40","SibSp":"1","Parch":"0","Ticket":"A/5 21171","Fare":"7.25","Cabin":"c","Embarked":"S"},
{"PassengerId":"1","Pclass":"3","Name":"Braund Mr. Owen Harris","Sex":"female","Age":"40","SibSp":"1","Parch":"0","Ticket":"A/5 21171","Fare":"7.25","Cabin":"c","Embarked":"S"},
{"PassengerId":"1","Pclass":"3","Name":"Braund Mr. Owen Harris","Sex":"male","Age":"40","SibSp":"1","Parch":"0","Ticket":"A/5 21171","Fare":"7.25","Cabin":"c","Embarked":"S"},
{"PassengerId":"1","Pclass":"3","Name":"Braund Mr. Owen Harris","Sex":"female","Age":"40","SibSp":"1","Parch":"0","Ticket":"A/5 21171","Fare":"7.25","Cabin":"c","Embarked":"S"},
{"PassengerId":"1","Pclass":"3","Name":"Braund Mr. Owen Harris","Sex":"male","Age":"40","SibSp":"1","Parch":"0","Ticket":"A/5 21171","Fare":"7.25","Cabin":"c","Embarked":"S"}]
```

## batch or manual scoring

- create a new notebook

```
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY
```

- Load the model

```
model = joblib.load('model.pkl')
```

- upload the json data file titanic4.json to jupyterlab directory

```
import pandas as pd
data = pd.read_json ('titanic4.json')
```

- Predict the output

```
result = model.predict_proba(data)
```

- Now convert the data to array 

```
df = data.to_numpy()
```

- Now display all the probablities

```
for i in range(len(df)):
    print("X=%s, Predicted=%s" % (df[i], result[i]))
```

## RESP API deployment

- Create a aks cluster
- Minimum of 12 cores is needed
- Deploy the automate ML model with custom configuration
- Score environment and score file are available in folder called RESTScoreFiles
- Deploy the rest service
- Will take few minutes
- Once deployment is succesful then test the services
- sample data

```
PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,3,Braund Mr. Owen Harris,male,22,1,0,A/5 21171,7.25,null,S
```

- the below image should show the probabilities

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AutoML/images/amlproba1.jpg "Service Health")