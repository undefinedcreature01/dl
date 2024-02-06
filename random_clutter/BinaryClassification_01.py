
import torch
from torch import nn

import sklearn 
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np


MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#create random seed
MANUAL_RANDOM_SEED = 12

torch.manual_seed(MANUAL_RANDOM_SEED)
torch.cuda.manual_seed(MANUAL_RANDOM_SEED)


#data
    #make 1000 samples

SAMPLE_N = 1000

#create circles

circles_dataset, circles_labels = make_circles(
    n_samples=SAMPLE_N,
    noise=0.03,
    random_state=MANUAL_RANDOM_SEED
)

circles = pd.DataFrame(
    {
        'x1':circles_dataset[:,0],
        'x2':circles_dataset[:,1],
        'label': circles_labels
    }
)

"""
#visualize

plt.scatter(
    x=circles_dataset[:,0],
    y=circles_dataset[:,1],
    c=circles_labels,
    cmap=plt.cm.RdYlBu #blue and red
)

plt.show()
"""

first_sample_data = circles_dataset[0]
first_sample_label = circles_labels[0]



#turn data into tensors + test split

circles_dataset = torch.from_numpy(circles_dataset).type(torch.float)
circles_label = torch.from_numpy(circles_labels).type(torch.float)


#random split

circle_train_data, circle_test_data, circle_train_labels, circle_test_labels = train_test_split(
    circles_dataset,
    circles_labels,
    test_size=0.2, # 20% - test; 80% train
    random_state=MANUAL_RANDOM_SEED
)

circle_train_data = circle_train_data.to(MODEL_DEVICE)
circle_train_labels = torch.tensor(circle_train_labels).type(torch.float).to(MODEL_DEVICE)

circle_test_data = circle_test_data.to(MODEL_DEVICE)
circle_test_labels = torch.tensor(circle_test_labels).type(torch.float).to(MODEL_DEVICE)


#model
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()

        #hidden layer
        self.linear_layer_1 = nn.Linear(
            in_features=2, #circle has 2 values (x1, x2)
            out_features=5 #one sample of x (2 feat) -> upscales to 5 (pre numbers to learn pattr. from)
        )

        #output layer ; in has to be same as the out of the prev. hidden layer
        self.linear_layer_2 = nn.Linear(
            in_features=5, #mathces with ll1 ; 
            out_features=1 #takes in 5 features -> downscales to 1
        )

    def forward(self, data):
        return self.linear_layer_2(self.linear_layer_1(data)) #data from layer1 -> layer2 -> output
    

#model with nn.Sequential ; step data through set layers
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=2,out_features=5),
            nn.Linear(in_features=5,out_features=1)
        )

    def forward(self, data):
        return self.linear_layers(data)
    

#acc ~ 50 % ; guessing even after 100 epochs ... ; fix the model

##added hidden units + added new layer ; achived nothing - not a linear problem
class CircleModelV3(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, data):
        return self.layer_3(self.layer_2(self.layer_1(data)))
    
#model with non linear act functions
class CircleModelV4(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

        self.layer_relu = nn.ReLU()

    def forward(self, data):
        return self.layer_3(self.layer_relu(self.layer_2(self.layer_relu(self.layer_1(data)))))


circle_model_0 = CircleModelV1().to(MODEL_DEVICE)
circle_model_1 = CircleModelV2().to(MODEL_DEVICE)
circle_model_4 = CircleModelV4().to(MODEL_DEVICE)

#calc accuracy ; what precentage does the model get right ? out of x samples
def accuracy_function(true, predictions):
    correct = torch.eq(true, predictions).sum().item() #just the amount
    accuracy = (correct/len(predictions))*100

    return accuracy



#training

#loss function for binary classif ? BECWithLogitsLoss
loss_function = nn.BCEWithLogitsLoss() #sigmoid built in

optimizer = torch.optim.SGD(
    params=circle_model_1.parameters(),
    lr=0.001
)


"""
train_logits = circle_model_1(circle_test_data.to(MODEL_DEVICE))[:5]

prediction_prob_labels = torch.sigmoid(train_logits) #between 0 and 1 ; need to round

# pred prov >= 0.5 class 1 ; pred prob < 0.5 class 0
train_predictions = torch.round(prediction_prob_labels)

"""



def train_loop(
        model,
        epochs,
        training_data,
        training_labels
):
    #training loop
    model.train()

    for epoch in range(epochs):

        #raw logits(output of a model) 
            #->(pass to act. fun) = prediction prob ->(round..) prediction labels

        #forward pass
        train_logits = model(training_data).squeeze()

        #turn logits into prob predictions ; what label(output) is the input ?
        train_predictions = torch.round(torch.sigmoid(train_logits))

        #loss ; BBCEWithLogitsLoss ; raw logits as input
        train_loss = loss_function(train_logits, training_labels)

        #accuracy
        train_accuracy = accuracy_function(training_labels, train_predictions)

        #optimizer ; reset
        optimizer.zero_grad()

        #backprop
        train_loss.backward()

        #step ; gradient descent
        optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch {epoch} | train loss {train_loss} train acc {train_accuracy}")

    return train_predictions

def test(
        model,
        test_data,
        test_labels
):
    #testing
    model.eval()

    with torch.inference_mode():
        test_logits = model(test_data).squeeze()
        test_predicitons = torch.round(torch.sigmoid(test_logits))
        
        #test loss
        test_loss = loss_function(test_logits, test_labels)

        #accuracy
        test_accuracy = accuracy_function(test_labels, test_predicitons)

    print(f" test loss {test_loss} | test acc {test_accuracy}")


print(train_loop(
    circle_model_0,
    1000,
    circle_train_data,
    circle_train_labels
)[:5])


"""
train_loop(
    circle_model_0,
    100,
    circle_train_data,
    circle_train_labels
)
"""