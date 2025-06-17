# MEG_model
This repo contains code for an ablation study for gauging the benefit of using self-supervised pretraining on MEG brain data. 

## Data and task

This MEG data is preprocessed to respect the temporal structure of the MEG nodes used for obtaining this data, that is: our data is a sequence of 2D MEG images. 

The task is to predict one of four activities based on the MEG data of the brain. 

## Model architecture

Our model architecture consists of a CNN that creates spatial features, which are then fed into an transformer sequence model, which is used for classification. This model is trained in an end-to-end fashion, but we investigate the benefit of self-supervised pretraining in this ablation study.

## Self-supervised pretraining

We want to study the effects of self-supervised pretraining, i.e. training tasks that use unlabeled data, with the intention of this helping the model generalize on the classification task. This way of training has been used to great effect in language tasks. The hope is that learning a representation of the data in this self-supervised fashion allows the model to generalize better to the classification task. 

### denoising autoencoder

For the CNN we investigate if it is beneficial to pretraining the model on a denoising task. To this end, we augmented the model with a decoder to form an autoencoder, and trained the model on reconstructing the clean MEG from noisy MEG data. During training, we got rid of the decoder but initialized the model using the weights learned in this reconstruction task. We hope that this makes the model robust to noise

### masked MEG node reconstruction

In another pretraining task, we again use the autoencoder setup but randomly mask nodes of the MEG data and task the model with reconstructing these nodes from the rest of the data. We hope that this forces the model to pick up spatial dependencies. 

### masked sequence reconstruction

For our transformer we mask parts of the sequence and train the model to reconstruct the missing part of the sequence. We hope that this encourages our model to learn context aware embeddings.

## ablation experiment

We train models on the classificaton task with each possible combinations of pretraining regimes, then evaluate the F1 score:

|   | Denoising | Masked MEG reconstruction | Masked sequence reconstruction | F1 |
|---|-----------|---------------------------|--------------------------------|----|
| 1 |           |                           |                                |    |
| 2 | x         |                           |                                |    |
| 3 |           | x                         |                                |    |
| 4 |           |                           | x                              |    |
| 5 | x         | x                         |                                |    |
| 6 | x         |                           | x                              |    |
| 7 |           | x                         | x                              |    |
| 8 | x         | x                         | x                              |    |


# Getting started

## Data
Download and unzip the data and move the `cross` and `infra` folder into  `data/files/`. then run `scripts/process_meg_data.py`, which downsamples and processes the meg data. 

## Poetry 
We make use of [Poetry](https://python-poetry.org/) to manage dependencies. If you install Poetry you can run each file from the root using.

```bash
poetry run python <directory>/<python file>.py
```
or

```bash
poetry run python -m <directory>.<python file>
```

if the file is a module

Using poetry is not strictly necessary, but if you don't you need to manage your own dependencies.

# subdirectories:

## data/ 
Here is where the (processed) data lives

`dataloader.py` contains data loaders for pretraining tasks as well as classification task training

## results/
Contains the stored locally obtained model weights in `model_weights` (not saved due to size) as well as plots and logs from training sessions in `runs`.

## scripts/
Contains a `process_meg_data.py`, which downsamples data and encodes spatial dependencies.

## training/
Contains scripts for pretraining the CNN and the sequence model, as well as `train.py` which trains a model end to end on the classification task.


