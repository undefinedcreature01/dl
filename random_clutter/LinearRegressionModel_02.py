#dependencies ...
import torch

from torch import nn as NEURAL_NETWORK #building blocks for neural networks/comp. graph

import matplotlib.pyplot as MAT_PLOT

import numpy as NP

import pathlib as PATH

import testing_data_LinearRegression as test_data


#model device

MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_data.test_reg_X.to(MODEL_DEVICE)
test_data.test_reg_y.to(MODEL_DEVICE) 

#create random seed
MANUAL_RANDOM_SEED = 42

torch.manual_seed(MANUAL_RANDOM_SEED)

#path 
MODEL_PATH = PATH.Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'linear_reg_model_0.pth'

MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


class LinearRegressionModelv2(NEURAL_NETWORK.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer = NEURAL_NETWORK.Linear(
            in_features=1, #get one feature
            out_features=1 #to output one feature
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(data)


lin_reg_model_2 = LinearRegressionModelv2()


