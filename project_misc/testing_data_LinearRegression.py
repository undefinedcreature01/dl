
import torch

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