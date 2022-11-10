# Adult Cencus Income End to End Machine Learning Project :moneybag:
This repository holds all the code, data, models, dependencies and deployment file, for the data cleaning and analysis, feature extraction, hyperparameter tuning, 
preprocessing, training, deployment and streamlit app of the Adult Census Income Dataset from the UCI Machine Learning Repository. 

This project is a mid-term project and is in partial fulfillment for the [mlbookcamp](https://datatalks.club/courses/2021-winter-ml-zoomcamp.html) course.  

# Table of Contents  
* [Description of the problem](#description)
* [Project Architecture](#architecture)
* [Model Deployment and Instructions](#deployment)

<a name="description"/>
### **Description of the problem :open_book:**

The aim of the project based on census data, is to predict if an individuals annual income exceeds $50,000 or not. The dataset used is referred to as 
the "Adult Census Income" dataset, which is very popular in the UCI Machine Learnig repository that can be found [here](https://archive.ics.uci.edu/ml/datasets/adult)
as well as in the data folder described in the [project architecture](#) below. Even though the census data is pretty outdated (1994), the solution
provided can produce insights on an individuls income and predict their income bracket.

<a name="architecture"/>
### **Project architecture: :triangular_ruler:**

```
├── Data
│   ├── Data Description      <- Contains the PDF for the data dictionary with explanation for each column.
│   ├── adult.data            <- Data file from the UCI machine learning repo that contains the training dataset.
|   └── adult.test            <- Data file from the UCI machine learing repo that contains the testing dataset.
│
├── images                    <- This folder contains the images that are used by and required for the Streamlit app.
│
├── models                    <- This folder contains the final trained model to be deployed using BentoML.
│
├── bentofile.yaml            <- This file is required by BentoML to build the "bento" i.e. service to deploy the model in the form of an API. It contains all the dependencies.
│
├── notebook.ipynb            <- This is the jupyter notebook where the data is loaded, cleaned, explored/analyzed and where feature extraction, hyperparameter tuning and model selection is done.
│
├── train.py                  <- This is the Python script that loads the data, does the preprocessing required for the final model chosen with its tuned hyperparameters and trains the model as well as save it using both BentoML and joblib.
│
├── predict.py                <- This is the Python script that loads the saved model and preprocessing transformer, creates the BentoML service and serves the API to finally make the prediction and return it
│
└── streamlit_app.py          <- This is the Python script that creates the streamlit which calls the model API, which is deployed on an Azure Container Instance and finally previews the final prediction in a user-friendly way
```

<a name="deployment"/>
### ** Model Deployment and Instructions :rocket:**

The model is deployed using [BentoML](https://www.bentoml.com/). The main script that is deployed is the predict.py. BentoML creates a bento from the 
predict.py and the bentofile.yaml files using ```bentoml build```. This bento is then containerized to a docker image using ```bentoml containerize```
This docker image is then pushed to Azure Container Registry, I chose Azure over other cloud providers as I already have student credits in my 
account from my university. I mainly used the Azure Command-Line Interface to accomplish the cloud deployment. 

* I firstly have to login uzing ```az login```
* Then I created an Azure container registry (acr) using ```az group create --name <aNameOfYourChoice> --location eastus`` choosing a location from ```az account list-locations```
* Then created an Azure container registry using ```az acr create --resource-group <aNameOfYourChoice> --name <acrName> --sku Basic```
* Then I logged in to the container registry using ```az acr login --name <acrName>```
* Then I tagged the container image using full name of the registry's login server which is found by ``` az acr show --name <acrName> --query loginServer --output table```
and we tag it like so: ```docker tag <dockerImageNameWithTag> <acrLoginServer>/<aNameOfYourChoice>```
* Then I pushed the image to Azure Container Registry using ```docker push <acrLoginServer>/<aNameOfYourChoice> ```. The image is now pushed after several minutes (it is 1.39GB)
* Then I deployed the container. First I need the registry credentials, I got them with ```az acr credential show --name <acrName>```, or in the Azure dashboard online.
Then I deploy the container with ```az container create --resource-group <aNameOfYourChoice> --name <aNameOfYourChoice> --image <acrLoginServer>/<aNameOfYourChoice> --cpu 1 --memory 1 --registry-login-server <acrLoginServer> --registry-username <service-principal-ID> --registry-password <service-principal-password> --ip-address Public --dns-name-label <aciDnsLabel> --ports <bentoMLPort>```

