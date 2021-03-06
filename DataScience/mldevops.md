# Why Datascience development is different from Software or Data/BI development

## Difference

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/DataScience/images/ds1.jpg "Service Health")

## Development

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/DataScience/images/mldevops-Development.png "Service Health")

- Bring production or actual data from various sources
- Data engineer the data set for use case
- Build a working data set for modelling
- Run data set through various algorithmns
- Need large compute depending on use case
- Compare the performance of model
- Iterate through the process to find the best model outcomes
- Once the algorithmn is found, Create Training script
- Create a model file
- Validate the model with new data set to test the performance
- if the performance is acceptable move forward, other wise go back to model development
- Once acceptable performance is good create score script for realtime or batch inference
- Create Evaluate or model compare script from previous run
- Scoring script to check for model performance to trigger new model build
- Save all the scripts in Github or some code repository
- Create documentation on use case, model and it's usage

## QA and Production

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/DataScience/images/mldevops-Production.png "Service Health")

- Get the code from code repository
- Create Azure DevOps or other pipeline tools to build the deployment process
- Run the Training process with production data
- Need large compute depending on use case
- Run testing of the model
- if the model performance is better than previous run then create Model brain file
- Use the Score file to create REST Api
- Create Docker container to run the Microservice
- Create the container orchestration enginer (if new), other wise use existing
- Deploy the new Rest API and decommision the old one (if exists)
- When score model is used if performance degrade create a model train run and deployment