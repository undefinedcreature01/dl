import torch
from torch import nn as NEURAL_NETWORK

import sklearn 
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

import pandas as PANDAS

import matplotlib.pyplot as MAT_PLOT

import numpy as NP

#create random seed
MANUAL_RANDOM_SEED = 42

torch.manual_seed(MANUAL_RANDOM_SEED)


#create circles
SAMPLE_N = 1000

DATASET_circles, LABELS_cricles = make_circles(
    n_samples=SAMPLE_N,
    noise=0.03,
    random_state=MANUAL_RANDOM_SEED
)

#turn from float64, int64 to tensors float32
DATASET_circles = torch.tensor(DATASET_circles, dtype=torch.float32)
LABELS_cricles = torch.tensor(LABELS_cricles, dtype=torch.float32)

#testing split ; "random"

TRAIN_DATA_circles, TEST_DATA_circles, TRAIN_LABELS_circles, TEST_LABELS_circles = train_test_split(
    DATASET_circles,
    LABELS_cricles,
    test_size=0.2, # 20% - test; 80% train
    random_state=MANUAL_RANDOM_SEED
)


#model

class CircleModel(NEURAL_NETWORK.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer1 = NEURAL_NETWORK.Linear(in_features=2, out_features=5)
        self.linear_layer2 = NEURAL_NETWORK.Linear(in_features=5, out_features=1)

    def forward(self, data):
        return self.linear_layer2(self.linear_layer1(data))
    

model_0 = CircleModel()

#training

loss_function = NEURAL_NETWORK.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(
                            params=model_0.parameters(), #set by the model itself
                            lr=0.01 #learning rate ; hyperparameter we can set it ; adjustemnt per loop
                            )




# Calculate accuracy - out of 100 examples, what percentage does our model get right? 
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item() 
  acc = (correct/len(y_pred)) * 100
  return acc
     

"""
#loop

epochs = 1000

# Build training and evaluation loop
for epoch in range(epochs):
  ### Training
  model_0.train()

  # 1. Forward pass
  y_logits = model_0(TRAIN_DATA_circles).squeeze()
  y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labels

  # 2. Calculate loss/accuracy
  # loss = loss_function(torch.sigmoid(y_logits), # NEURAL_NETWORK.BCELoss expects prediction probabilities as iNPut
  #                TRAIN_LABELS_circles)
  loss = loss_function(y_logits, # NEURAL_NETWORK.BCEWithLogitsLoss expects raw logits as iNPut
                 TRAIN_LABELS_circles)
  acc = accuracy_fn(y_true=TRAIN_LABELS_circles, 
                    y_pred=y_pred)
  
  # 3. Optimizer zero grad
  optimizer.zero_grad()

  # 4. Loss backward (backpropagation)
  loss.backward()

  # 5. Optimizer step (gradient descent)
  optimizer.step() 

  ### Testing
  model_0.eval()
  with torch.inference_mode():
    # 1. Forward pass 
    test_logits = model_0(TEST_DATA_circles).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))

    # 2. Calculate test loss/acc
    test_loss = loss_function(test_logits,
                        TEST_LABELS_circles)
    test_acc = accuracy_fn(y_true=TEST_LABELS_circles,
                           y_pred=test_pred)
  
  # Print out what's happenin'
  if epoch % 10 == 0:
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

"""


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = NP.meshgrid(NP.linspace(x_min, x_max, 101), NP.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(NP.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    MAT_PLOT.contourf(xx, yy, y_pred, cmap=MAT_PLOT.cm.RdYlBu, alpha=0.7)
    MAT_PLOT.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=MAT_PLOT.cm.RdYlBu)
    MAT_PLOT.xlim(xx.min(), xx.max())
    MAT_PLOT.ylim(yy.min(), yy.max())

"""
# Plot decision boundary of the model
MAT_PLOT.figure(figsize=(12, 6))
MAT_PLOT.subplot(1, 2, 1)
MAT_PLOT.title("Train")
plot_decision_boundary(model_0, TRAIN_DATA_circles, TRAIN_LABELS_circles)
MAT_PLOT.subplot(1, 2, 2)
MAT_PLOT.title("Test")
plot_decision_boundary(model_0, TEST_DATA_circles, TEST_LABELS_circles) 
     
MAT_PLOT.show()
"""

class CircleModelV2(NEURAL_NETWORK.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = NEURAL_NETWORK.Linear(in_features=2, out_features=10)
    self.layer_2 = NEURAL_NETWORK.Linear(in_features=10, out_features=10)
    self.layer_3 = NEURAL_NETWORK.Linear(in_features=10, out_features=1)
    self.relu = NEURAL_NETWORK.ReLU() # relu is a non-linear activation function
    
  def forward(self, x):
    # Where should we put our non-linear activation functions?
    return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CircleModelV2()

# Setup loss and optimizer
loss_fn = NEURAL_NETWORK.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(), 
                            lr=0.1)


# Loop through data
epochs = 10000

for epoch in range(epochs):
  ### Training
  model_3.train()

  # 1. Forward pass
  y_logits = model_3(TRAIN_DATA_circles).squeeze()
  y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels

  # 2. Calculate the loss
  loss = loss_fn(y_logits, TRAIN_LABELS_circles) # BCEWithLogitsLoss (takes in logits as first input)
  acc = accuracy_fn(y_true=TRAIN_LABELS_circles,
                    y_pred=y_pred)
  
  # 3. Optimizer zero grad
  optimizer.zero_grad()

  # 4. Loss backward
  loss.backward()

  # 5. Step the optimizer
  optimizer.step()

  ### Testing
  model_3.eval()
  with torch.inference_mode():
    test_logits = model_3(TEST_DATA_circles).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    
    test_loss = loss_fn(test_logits, TEST_LABELS_circles)
    test_acc = accuracy_fn(y_true=TEST_LABELS_circles, 
                           y_pred=test_pred)
  
  # Print out what's this happenin'
  if epoch % 100 == 0:
    print(f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
     

print(test_pred[:5])
print(TEST_LABELS_circles[:5])