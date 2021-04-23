# Readme
This is final project for CS227: Managing and Mining Massive Time Series.

## File explanation
single_experiment.py: This file contains a single experiment. It also contains basic data augmentation methods. It allows to easily change the dataset name ane other hyperparameters and save the output model to specified address. Then it will load the saved model to run sample_evaluation that provided by stencil code. Most of this code is from provided stencil code.
auto_encoder.py: this file contains our main model. It has autoencoder (setting similarity to 0 can make it the same as baseline model), autoencoder with similarity encoder model. We need to manully change step function. train_step is for autoencoder, train_step_v3 is for autoencoder with similarity encoder.
sample_evaluation.py: This file contains evaluation code that load saved model and then run evaluation respecting reconstruction loss, MAE, common nn ED, common nn SBD(needs to edit code a little bit).
run_experiment.py: This file contains code to specify what experiments need to run and put the evaluation results in a specified CSV documents. There are different experiments in this file such as epochs, similarity, filter sizes and different datasets.
evaluation_utils.py: This file contains summarized evaluation utilities functions. 
timeseries-similarity.ipynb: This file contains python notebook file that visualizes and evaluates results we collect from CSV files.


## How to use

1. Download and unpack the file [here](https://drive.google.com/file/d/13PwgJNBTnyT1IjbUxFqQlqq2VTGDVw8N/view?usp=sharing) to the `data/` directory
2. To run a sample evaluation: `python3 sample_evaluation.py -d GunPoint`
