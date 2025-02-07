# Exploring the Weight Space of Neural Networks for Learning and Generation

This is the code repository of my Master Thesis in Saarland University on "Exploring the Weight Space of Neural Networks for Learning and Generation" topic. 
There are two main objectives in this project. 
First is to explore the weight space of the classifier MLP zoos and train other models to classify the features of the dataset used to train these zoos.
The model architectures used to train parameter classifiers are: MLP, [Set Transformers](https://arxiv.org/pdf/1810.00825), [DWSNets](https://arxiv.org/pdf/2301.12780).
Second is to train a generational models in the weight space of these zoos to conditionaly generate new set of parameters.
We compare the generated parameters between Variational Autoencoder, Diffusion and Condition Flow Matching models.

## Structure

* src - contains the reusable code used throughout the research.
* data - should contain dataset files for each zoo and data splits. Left empty becuase dataset files are too large for github. The files can be found in [Google Drive](https://drive.google.com/drive/folders/1ANk3a5drWipgdUFWfBeAfyfN23nSEFXu) and put into the folder manually.
* configs - contains the .yaml config files for the MLP classifiers.
* reports - contains plots and diagrams with results.
* notebooks - contains jupyter notebooks used to train and evaluate classifier and generative models.

### Interesting notebooks:

* dataset.ipynb - contains code used to train the model zoos and their conversion into a dataset format.
* classification.ipynb - contains the evaluation of classifier models.
* generation.ipynb - contains the evaluation of generational models.
* models folder - contains notebooks with training code for all the models (except model zoos, look dataset.ipynb) used in this project.

## Parameter classification accuracies

<p align="center">
  <img src="https://github.com/IDavron/exploring-neural-networks-weight-space/blob/main/reports/big-zoo-classification-accuracy.png" width=500>
</p>

## Generated decision boundary examples

<img src="https://github.com/IDavron/exploring-neural-networks-weight-space/blob/main/reports/sample-1.png" width=400> <img src="https://github.com/IDavron/exploring-neural-networks-weight-space/blob/main/reports/sample-2.png" width=400>
