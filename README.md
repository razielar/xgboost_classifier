# XGBoost Classifier

* Data obtained from this [paper](https://www.science.org/doi/10.1126/science.aah7111).   

In this repository, we trained a XGBoost classifier modifying its `cost-function` to improve accuracy, due high data imbalance. 

The final model is located: `./jp_nbs/model` and saved as a pickle file. 

The Docker image is hosted in [Docker Hub]().

## Future work

Serve the model using `TensorFlow Serving` and `FastAPI` to perform **batch inference**. Then, create a Kubernetes cluster to orquestate the services. 

