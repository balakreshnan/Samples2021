# Azure Data Factory - Access Microsoft Graph API

## Use Service Principal to access Microsoft Graph API

## Steps

## Create a Service principal account

- Create a new service principal
- https://docs.microsoft.com/en-us/azure/azure-sql/database/authentication-aad-service-principal-tutorial
- Assign Microsoft Graph -> Users -> Users.Read.All
- Create a secret
- Store the Client id and secret in Azure Keyvault

## Azure Data Factory 

- Entire flow to view all users in microsoft graph api

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/graphapi1.jpg "Service Health")

- First we need to bring client id and secret from Key vault
- use Web Acitivity to get the secrets and assign to variables
- Create a new pipleline
- Create variables called
    - clientid
    - clientsecret
    - token
- now log into azure portal
- Go to Azure keyvault
- go to secrets and copy the URL to access
- Make sure ADF managed identity has contributor in keyvault
- Also provide get permission to read the keys in azure keyvault
- Now lets get the client id
- Drag the web activity

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/graphapi2.jpg "Service Health")

- Assign the output to variable

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/graphapi3.jpg "Service Health")

- Lets bring client secret now

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/graphapi4.jpg "Service Health")

- Assign to clientsecret variable

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/graphapi5.jpg "Service Health")

- Now time to get the authoriazation token to use for microsoft graph

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/graphapi6.jpg "Service Health")

- Assign the token to variable called token

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/graphapi7.jpg "Service Health")

- Now call the Microsoft graph to retrieve data
- Need the token for authorization

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/ADF/images/graphapi8.jpg "Service Health")