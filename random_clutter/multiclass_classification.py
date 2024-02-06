
#DEPENDECIES ------------------------------------------

#deep learning help
import torch as torch
from torch import nn as NEURAL_NETWORK

import sklearn 
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

#utility
import numpy as NP
import pandas as PD

#visual
import matplotlib.pyplot as PLT

#constants

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 21

MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLOB_DATASET, BLOB_LABELS = make_blobs(
    n_samples=1000,
    n_features=NUM_FEATURES,
    centers=NUM_CLASSES,
    cluster_std=1.5, 
    random_state=RANDOM_SEED)

#turn data into tensors 
BLOB_DATASET = torch.from_numpy(BLOB_DATASET).type(torch.float32)
BLOB_LABELS = torch.from_numpy(BLOB_LABELS).type(torch.float32)

#train + test sets

TRAIN_BLOB_DATASET, TEST_BLOB_DATASET, TRAIN_BLOB_LABELS, TEST_BLOB_LABELS = train_test_split(
    BLOB_DATASET,
    BLOB_LABELS,
    test_size=0.2,
    random_state=RANDOM_SEED
)


#DEFINITONS ------------------------------------------

class BlobModel(NEURAL_NETWORK.Module):
    def __init__(self, _input_features, _output_features, _hidden_units=8):
        super().__init__()

        self.linear_layer_stack = NEURAL_NETWORK.Sequential(
            NEURAL_NETWORK.Linear(in_features=_input_features, out_features=_hidden_units),
            #NEURAL_NETWORK.ReLU(),
            NEURAL_NETWORK.Linear(in_features=_hidden_units, out_features=_hidden_units),
            #NEURAL_NETWORK.ReLU(),
            NEURAL_NETWORK.Linear(in_features=_hidden_units, out_features=_output_features)
        )

    def forward(self, data):
        return self.linear_layer_stack(data)


#calc accuracy ; what precentage does the model get right ? out of x samples
def accuracy_function(true, predictions):
    correct = torch.eq(true, predictions).sum().item() #just the amount
    accuracy = (correct/len(predictions))*100

    return accuracy



#VARS ------------------------------------------

model_4 = BlobModel(
    _input_features=2,
    _output_features=4,
    _hidden_units=8
)


LOSS_FUNCTION = NEURAL_NETWORK.CrossEntropyLoss()

OPTIMIZER = torch.optim.SGD(
    params=model_4.parameters(),
    lr=0.1
)


#loop vars

epochs = 100 #cycles trhough all data

logits = None #raw output of a model 

label_propability_predictions = None #logits + some activation function ; 

label_final_predictions = None #prediction probability + some function(depends on type)

train_loss = None #float ; avg diff between ideal and model output

train_accuracy = None #proc of how accurate it is based on x semples

#exe ------------------------------------------

"""
model_4.eval()

with torch.inference_mode():
    logits = model_4(BLOB_DATASET)

#softmax ; all values in a line - sum up to 1
    #each value in logits line = one class ; % of it being that class 
    # -> we need the index of the max value (predicted class)

label_propability_predictions  = torch.softmax(logits, dim=1) #activations
label_final_predictions = torch.argmax(label_propability_predictions , dim=1) #take the argmax
"""


#training loop

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

#put data to target device
TRAIN_BLOB_DATASET = TRAIN_BLOB_DATASET
TRAIN_BLOB_LABELS = TRAIN_BLOB_LABELS.type(torch.LongTensor)

TEST_BLOB_DATASET = TEST_BLOB_DATASET
TEST_BLOB_LABELS = TEST_BLOB_LABELS.type(torch.LongTensor)


for epoch in range(epochs):
    model_4.train()

    logits = model_4(TRAIN_BLOB_DATASET)

    #preform act on raw logits -> final func on that
    label_final_predictions = torch.softmax(logits, dim=1).argmax(dim=1)

    train_loss = LOSS_FUNCTION(logits,TRAIN_BLOB_LABELS)

    train_accuracy = accuracy_function(TRAIN_BLOB_LABELS, label_final_predictions)

    OPTIMIZER.zero_grad()

    train_loss.backward()

    OPTIMIZER.step()

    #testing
    model_4.eval()

    with torch.inference_mode():
        test_logists = model_4(TEST_BLOB_DATASET)
        test_final_predictions =  torch.softmax(test_logists, dim=1).argmax(dim=1)
        test_loss = LOSS_FUNCTION(test_logists, TEST_BLOB_LABELS)
        test_accuracy = accuracy_function(TEST_BLOB_LABELS, test_final_predictions)
        

