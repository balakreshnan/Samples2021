# Azure ML Ops using github actions

## MLOps using Github Actions

- Need Github Account
- Create a Azure Machine learning workspace
- Create a compute
- Create a service principal
- Store the secret in github secrets
- Clone the repo change settings in workspace.json
- Change the compute instance to match what you created in compute.json
- In train/run_config.yml change the target: {name of the compute created}
- save and push to trigger run
- Go to github actions to see how it executes the CI/CD

- Follow the tutorial from this page

https://docs.microsoft.com/en-us/azure/machine-learning/how-to-github-actions-machine-learning

https://techcommunity.microsoft.com/t5/azure-ai/using-github-actions-amp-azure-machine-learning-for-mlops/ba-p/1419027