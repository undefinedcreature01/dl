
#dependencies ...
import torch

from torch import nn as NEURAL_NETWORK #building blocks for neural networks/comp. graph

import matplotlib.pyplot as MAT_PLOT
import numpy as NP
import pathlib as PATH

#create random seed
MANUAL_RANDOM_SEED = 42

torch.manual_seed(MANUAL_RANDOM_SEED)

#path 
MODEL_PATH = PATH.Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'linear_reg_model_0.pth'

MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


#starts with random par ; update the numbers based on forward -> with gradient descend

# "simple straight line"
class LinearRegressionModel(NEURAL_NETWORK.Module): #pytorch models are a subclass of this ; req_grad = true / default
    def __init__(self):
        super().__init__()

        self.weight = NEURAL_NETWORK.Parameter(torch.rand(1,requires_grad=True,dtype=torch.float))
        self.bias = NEURAL_NETWORK.Parameter(torch.rand(1,requires_grad=True,dtype=torch.float))
        
    #forward (computation) method ; required(nn)/override
        #operation that the model does (core here: linear ref. function)
    def forward(self, data:torch.Tensor) -> torch.Tensor:
        return self.weight * data + self.bias
        
linreg_model = LinearRegressionModel()
    #check contents ; print(linreg_model.state_dict()) #named parameters + value


#testing data

IDEAL_WEIGHT = 0.7
IDEAL_BIAS = 0.3

START = 0 #starting value
END = 1 #end value (last value in array is end - step)
STEP = 0.02 #how fast we reach the value ; eg d between values


#all the data
tensor32_x = torch.arange(START,END,STEP).unsqueeze(dim=1) #unsqueeze ; each value array of their own

tensor32_y = IDEAL_WEIGHT * tensor32_x + IDEAL_BIAS #lin reg formula

#splitting testing data 80 train - 20 test

#we want the model to : get x - return y
point_to_split = int(0.8 * len(tensor32_x)) #80 % of data -> training set/split

train_reg_X = tensor32_x[:point_to_split] #everything before the split
train_reg_y = tensor32_y[:point_to_split]

test_reg_X = tensor32_x[point_to_split:] #everything after the split (onwards)
test_reg_y = tensor32_y[point_to_split:]


#visualize

def plot_predictions(
                    train_data,
                    train_labels,
                    test_data,
                    test_labels,
                    predictions=None):
    
    MAT_PLOT.figure(figsize=(10,7))

    if train_data is not None:
        #training data - blue
        MAT_PLOT.scatter(train_data, train_labels, c='b', s=4, label="training data")

    if test_data is not None:
        #test data - green ; what we are looking at
        MAT_PLOT.scatter(test_data, test_labels, c='g', s=4, label="testing data")


    #are there predictions 
    if predictions is not None:
        #predictions - red; what we got from the model ; ideal green = red
        MAT_PLOT.scatter(test_data, predictions, c='r', s=4, label="predictions")

    #legend
    MAT_PLOT.legend(prop={"size":14})

    MAT_PLOT.show() #opens a seperate window ..



#plot_predictions(train_reg_X, train_reg_y, test_reg_X, test_reg_y)



#get predictions / testing the model
    
"""
with torch.inference_mode(): #context managerr ; inference = prediction mode ; not saved to memory
    predictions_y = linreg_model(test_reg_X)


print(predictions_y) #our predictions
print(test_reg_y) #what its supposed to be (ideal)

plot_predictions(train_reg_X, train_reg_y, test_reg_X, test_reg_y, predictions=predictions_y)
"""

#when we just run it like this : random values + no code : bad predicitions
    #to evaluate our model : loss function (how good/bad the predictions are)
    #absolute error = mean error


# specific functions to use : depends on the problem
    # ex: reggression problem

loss_function = NEURAL_NETWORK.L1Loss()


optimizer = torch.optim.SGD(
                            params=linreg_model.parameters(), #set by the model itself
                            lr=0.01 #learning rate ; hyperparameter we can set it ; adjustemnt per loop
                            )


#training + testing loop

#loop through data
epochs = 100

epoch_count = []
train_loss_values = []
test_loss_values = []

for epoch in range(epochs):

    #model - set training moed
    linreg_model.train() #all par that req grad - true

    #forward pass
    predictions_x = linreg_model(train_reg_X)

    #calc loss
    loss = loss_function(predictions_x, train_reg_y) #this should be 0

    #optimizer zero grad
    optimizer.zero_grad() #optimizes accumulates on each loop ; each it : fresh

    #preform backprop (from output - back ; red grad) ; backwards pass
    loss.backward()

    #preform gradient descent ; adjust - to get to the bottom (loss grad = 0)
    optimizer.step()


    #testing ------------------------------------------------------
    linreg_model.eval() #turn off grad tracking .. stuff not needed for evaluating
    
    with torch.inference_mode(): #context managerr ; inference = prediction mode ; not saved to memory
        test_predictions = linreg_model(test_reg_X) #data it has never seen

        test_loss = loss_function(test_predictions, test_reg_y)
    
    epoch_count.append(epoch)
    train_loss_values.append(loss)
    test_loss_values.append(test_loss)



#visualize the curves

MAT_PLOT.plot(
            epoch_count, 
            torch.tensor(test_loss_values).numpy(),
            label="test loss")

MAT_PLOT.plot(       
            epoch_count, 
            torch.tensor(train_loss_values).numpy(),
            label="train loss")

MAT_PLOT.title("train + test loss curve")

MAT_PLOT.ylabel("loss")
MAT_PLOT.xlabel('epoch')

#legend
MAT_PLOT.legend(prop={"size":14})

MAT_PLOT.show() 



#saving the model
    #torch.save # save object in pickle format
    #torch.load
    #torch.nn.Module.load_state_dict $allos to load a models saved state

#save the model state dict (not the entire model)
torch.save(obj=linreg_model.state_dict(), f=MODEL_SAVE_PATH)


#create new object
loaded_lin_reg_model_9 = LinearRegressionModel()

#load saved data
loaded_lin_reg_model_9.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

print(linreg_model.state_dict())
print(loaded_lin_reg_model_9.state_dict())

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available.")