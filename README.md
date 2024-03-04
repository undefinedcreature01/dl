


# 4. predmet ; izdelek - DEEP LEARNING


# What even is artificial intelligence ?

**Artificial intelligence** is a branch of computer science focused on developing and reaserching systems made to preform tasks that require human-like intelligence. 
<br>(the term "AI" may also be used to describe such machines/software). 

AIs are generally quite specialized, often designed to preform only one task(or a small range of tasks). For example just recognizing emotions or just recognizing shapes. We call such specilized AIs **narrow or "weak" AIs.**

(opposite: general AIs ; able to learn, understand and adapt - learn from experience, adapt to new/different situations ... currently only a goal in the field of AI)

# Machine learning

> Machine learning is a subfield of AI, that achives the preforming of tasks requring human-like intelligence with algorithms, creating models to preform the tasks without directly programming them to do so. 

AI is all about making systems able to preform tasks that generally require human-like intelligence right ? 

So machine learning is focused on achiving that with the development of algorithms and models that enable systems to preform those tasks without being programmed with specific rules on how to preform them, but rather by learning on specific datasets and applying that to new data.

Generally, they are used when:

- the solution for the task at hand is not clear/direct/simple,

- there is repeated decision making/evalution

- we have labled data, mapped to the desired result

| |   |
|---|---|
| procedure | a process or set of rules to follow  |
| program | a series of coded software instructions  |
| algorithm  | a process or set of rules to follow in order to solve a problem |
| method | a series of actions conducted in a certain order or manner in order to solve a problem |
| |   |

>
>As with (apparently) everything withing the field of AI many terms are used interchangeably.
>
>So in general algorithm = method = program (all procedures; series of actions to do something)
>
>Model is a program - programs are algorithms - algorithms are methods, so models are also called methods.
>
>Methods withing machine learning are also just procedures, that tell the model what to do with the data. (they are algorithms - still part of a model, making them methods withing a method if you will)
>
>Let's draw a very clear line on what each thing is
>
>| |   |
>|---|---|
>| model | is the program that we give data to in order to preform the task that requires human-like intelligence. <br> (want to predict housing prices based on house size ? - you create a model to do it) |
>| method | procedures to process data withing a model <br> (they process the housing size-price data ; there can be many within a model) |
>| |   |

</span>
<br>


| |   |
|---|---|
| training algorithm  | specific procedure/method/algorithm  used to adjust the parameters of a machine learning model |
| dataset | collection of data  |
|  labled data | label-data pairs where the label corresponds to the data |
| label | the desired output of a system in a labled dataset (ie. answer, category) |
| features  | relevant/useful information about/from a specific instance of data (ie. edges of a picture, pitch of a sound, results of a function)|
| |   |

## Models

> Models in machine learning are programs, trained with a training algorithm on some training or test data - and are then able to take new unseen data an make predictions. 

Tweaked values/patterns inside a model that help it predict/preform the task with given data are the end result of training with an algorithm.

The structure/architecture of a model differs based on the task it is trying to preform and the complexity of such task. 

| |   |
|---|---|
| regression |  method used to model the relationship between a dependent variable (ie. cost of a house) and one or more independent variables(ie. size of a house), aiming to predict the value of the dependent variable based on the values of the independent variables |
| classification |   |
| |   |

|Model architectures||
|-|-|
| Regression models | |
| Decision tree |  |
| (Artificial) Neural network - (A)NN| model inspired by the biological neural network <br> - connections of neurons data travels through|

## Methods

Methods are procedures for accomplishing or approaching something - generally a goal. 
<br> In machine learning methods are both parts of a training algorithm used to train a model, and parts of its structre. 

Learning paradigms generally tell us how a model interacts with data during training. Do we give it labeled data and tell it what things are ? Should the model discover patterns by itself ? And so on.

| Learning paradigms|  |
|-|-|
| Supervised learning      | training algorithm uses labled training data (tries getting closer to the correct label)                  |
| Unsupervised learning    | training algorithm uses unlabled training data - discoveres patterns by itself <br> (usually based on the difference between input and output)                              |
| Semi-Supervised learning | training algorithm uses both labled and unlabled data <br> (usually used when labled data is scarce/expensive)             |
| Reinforcement learning   | training algorithm trains by interacting - receiving feedback (rewards/penalties) 
| Self-supervised learning   | |

There are types of methods that utlize learning paradigms, and do not fit very well into the categorization above. They generally tell us what else the model does with the data given - be it labled, unlabled or something in between.

|||
|-|-|
| Feature/representation learning      | model automatically identifies/extracts features from raw data <br> (unsupervised, supervised)     
|Association rule learning| model identifies 'rules' to store - to process data with <br> IF data_is_something THEN output |
| |   |

## (Artificial) Neural Networks (ANNs)

Artificial neural networks are machine learning models, inspired by the biological neural network.

The layred structure comes in handy when we have large amounts of data that needs to be processed.

| |   |
|---|---|
| neuron/node/unit | cells, that recieve signal, process it and then forward it  |
| layer |  building block, groups up nodes that preform specific operations on data |
| weight | how important a value from a neuron has <br> (generally multiplied - so if a weight is really low the output value will also be low - having less impact)  |
| treshold| a point something has to cross |
| link/connection |   |
| |   |

In the artificial neural networks neurons are also referred to as nodes or units. They take in data either from some external input data (ie. we give it) or from other nodes in the network, process it (run the given data through a function/method that returns a new value of the neuron) and produce an output - that is then sent to other nodes or treated as the final output.

Neural networks are models made up of nodes, that are connected together. These nodes are usually combined into layers. 

| |   |
|---|---|
| input layer | layer that recieves external data  |
| output layer | layer that produces the final result  |
| hidden layer(s) | all layers (0+) between the input layer and the output layer of a neural network   |
| |   |

The nodes can connect to the following layers in differnt ways:

| Connection patterns/direction |   |
|---|---|
| fully connected |  every node of one layer - connects to every node of the next layer |
| pooling | group of nodes in one layer - connect to one node in the next layer |
| reccurent | nodes can connect to nodes in the next, its own or the previous layer |

When a neural network has 2 or more layers, it falls into the category of DNNs - deep learning networks.

We have two borad types of ANNs. They define how the data flows trough the mode.

- FNNs (feedforward) 

Data flows only in one direction - forward. 
<br>(from the input layer - trough the hidden layers(if there are any), to the output layer). Here each node has a weight
>A very common type of FNN is CNN (Convolutional neural network)

- RNNs (reccurent) 

Here nodes aren't independent blocks. The outputs of nodes (stored in memory) influance the outputs of other nodes (during the execution). This comes in handly when handling sequential data (ie. language; speech/writting).

Nodes in RNNs don't each have their own weight, but rather nodes inside each layer share the same parameters.

### Nodes - in FNNs

Nodes generally have a weights, these values represent the "strenght" or importance of a connection between neurons in adjescent layers. 

>
>A node sends its ouput to a node (or many nodes) in the next layer through a connection/link. There exist one connection for each node it sends its output to.
(connection is a pair; current node - next node)
>
>The weight on the connection between the node sending its output and the node recieving it tell the recieving node how important the value is in calculating its own output.

Each input value a node recives is multiplied by the weight of the connection it recieved it from, getting a "weighted input". (w = weight, x = input from a node)

$$f(x) = w * x$$

All the weighted inputs are then summed up. (n = number of nodes sending their inputs to the node that is calculating the sum)

$$ sum = (w_1 * x_1 + w_2 * x_2 + w_n * x_n)$$

The summed up weighted inputs are then sent to an activation/transfer function and this transformed input (output of a node) is then sent onward. 

$$y = activation\_function(w_1 * x_1 + w_2 * x_2 + b)$$

Nodes (can) also have something called **bias term/unit** (b in the above equation). It is a parameter(means the value changes when the model is trained) associated with each node in an network used to shift/offset the weighted value of the input. Shoft left to delay or right to accelerate the activation of a node.

Being able to shift the function result means it is not stuck at [0,0] like most functions are. 
<br> So the output of a node is not strictly defined, it is also a parameter that isn't input dependant (like weight is)

So each node is characterized by a weight, a bias and an activation function.
At the start(when a model is first created) these weights and biases are just some random numbers. Throught training a model tweaks these paramters in a way where it is then able to calculate the desired output from given inputs.

| |   |
|---|---|
| features  | relevant/useful information about/from a specific instance of data (ie. edges of a picture, pitch of a sound, results of a function)|
| in-features (input) |  relevant information a node recieves |
| out-features (output) |  relevant information a node outputs |
| |   |

Till now I have been referring to the data sent/recieved by a node as just input/ouput. What nodes actually recieve and output are features, relevant information from the data they recieve. 

### Deep learning

| |   |
|---|---|
| logits |  raw output of a model (values before an activation function is preformed) |
| |   |

Deep learning models are neural network models with more than two layers. 
(input, atleast one hidden layer and output). They use feature learning.

### Activation functions

Activation functions help map out data the nodes recieve in a way where it makes sense for fruther calculations. 

> (instead of values like 1.3 or 34(raw data - logits) we can use an activation function to get a value from 0 to 1 - which makes things much more clear)

> [! code and visualization here !](notebooks/00_activation_functions.ipynb)

| Function |   | Formula |
|---|---| --- |
| binary step | checks if a treshold was reached or not  | $$ f(x) = \begin{cases} 0 & \text{if } x < \theta \\ 1 & \text{if } x \geq \theta \end{cases} $$ |
| sigmoid | maps values as probabilities (between 0 and 1)  | $$\sigma(x) = \frac{1}{1 + e^{-x}}$$ |
| tanh | sigmoid - but the range is form -1 to 1  |$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$|
| ReLu(Rectified Linear Unit) |   | $$\text{ReLU}(x) = \max(0, x)$$ |
| Leaky ReLu|||
| ELU ||

# Training models ?

Training a model means adjusting weights and biases so that when it recieves data (unseen ; was not part of training) it can accuretly preform the desired task (ie. correctly classifying what is on an image..)

When training deep neural network models the most common talked about concept is back-propagation. 

| |   |
|---|---|
| epoch | one pass trough a model  |
| prediction | output generated by a model when given input data  |
| loss/error/cost | difference between the output a model generates and the desired output |
| |   |

## Backpropagation

Backpropagation is a way to train an FNN based model. This is a gradient based method - it adjusts the paramters of a model based on the error/loss between models predictions(its own output) and the actual desired output 
> It tries to find the minimum of a loss function, which is a concept known as gradient descent.

Backpropagation consists of two phases. 
- forward propagation, where the data is fed forward. 
- backward propagation, where data goes back from the output layer to the input layer, adjusting paramters along the way 

Feedforward is not strictly related to training. It simply tells us how data moves through the model. It refers to the action of data being "fed" chronologically from the input layer, trough the hidden layers(if there are any), to the output layer.

This is used both when a model is training and when it is being used to preform a task.

When training we generally use a "training loop". We do a pass trough a model (often reffered to as an epoch) consisting of these steps:

1. feedforward - get predictions (what our model thinks the correct output is)
2. compare them to what they actually should be - calculate the loss (which should be 0) with a loss function (ie. we check how our model preformed)
3. send the loss to an optimizer function that adjusts the parameters

> While backpropagation is the dominant method of training a neural network model there are other training algorthims. 

| |   |
|---|---|
| mean |   |
| entropy | indicates the amount of uncertainty of an event |
| ground truth | the desired output (what something actually is) - the true value (not a prediction)  |
| |   |

| Mathematical symbol |   |
|---|---|
|$n$ | number of value pairs/samples  |
| $y$ = $t$ = $T$ | the ground truth/truth label |
| $p(y)$ = $p$ = $\hat{y}$ = $s$ | models prediction (usually between 0 and 1) |
| $f()$ | activation function - transforms output|

> many symbols are used for the same terms - for the sake of clarity in this file 
> - $T$ .. truth
> - $p$ .. prediction

## Loss/cost functions / Cirterion

These functions tell us how far off the models predictions are - compared to what they should be, so they compare the difference between the prediction and the desired output. (ideal loss is then 0 - they should be the same)
>They (as other functions inside a model) are very task/problem dependant.

> [! code and visualization here !](notebooks/01_loss_functions.ipynb)


### Mean Squared Error (MSE) - L2/Quadratic Loss

> commonly used in regression tasks (ie. linear graphs)

We calculate the mean (average) of the squared difference between each prediction and the desired ouput. Wrong predictions are punished more because the error is squared (so if the difference is big - it squared is even bigger)

$$MSE(T_i, p_i) = \frac{1}{n} \sum_{i=1}^{n} (T_i - p_i)^2$$

So a loss for each example/pair of label - prediction is :

$$MSE(truth, prediction) = (truth - prediction)^2$$

### Mean Absolute Error (MAE) - L1 Loss

### Cross-Entropy  
> commonly used in classification tasks

This is used when we are "guessing" if something belongs in a certain class or not. 

So the question is "how sure are you that the instance belong in (x) class ?" - and generally the asnwers are between 0 and 1 (like procenteges are between 0% and 100%). So if im very sure i'd say 99% - which as a prediction would map to 0.99 

>(this is what activation functions do - take in some input and output it in a useful way).

| |   |
|---|---|
| $C$ | often used as number of classes |
| |   |

Logarithm (log) is a function sensitive to differences. What that means is that it will react greatly weather a prediction is 0.1 or 0.9. 

$log(0.1)≈$ (big loss)

$log⁡(0.9)≈ $ (small loss)

Since in cross-entropy the final loss is the sum of all losses (divided by the number of samples) we want those losses to be as small as possible - meaning whatever is sent to the log function should be a high value.

> When the number of classes is more than two(binary classification) it is a multi-class classification.

#### **Binary Cross-Entropy - Log Loss**
> commonly used in binary classification tasks (1 or 0 ?)

$C' = 2$, where : $C_1 = 1$ and $C_2 = 0$ 

| |   |
|---|---|
| positive class | arbitrary.. (usually just 1), the "more significant" class ; this is what the model is referencing    |
| negative class | the other class (usually 0)  |
| |   |

Here when asking the question "how sure are you that the instance belong in (x) class ?" the x class is the positive class, if we don't we run into a small problem.

> "how sure are you that the instance belongs in **class 1** ?"
<br> - very! 99% 
<br>"how sure are you that this new instance belongs in **class 0** ?" 
<br> - highly ! 89% !

What is the problem here ?

So for the sake of easier understanding let's say we are predicting weather there is a cat(1) or a dog(0) on an image.

We give a model a picture and it returns 0.9. What does that tell us ? Nothing because this picture doesn't have a label yet (this is what the model is for). We don't know what is on the picture so how could we ask a question in the format of "how sure are you that the instance belongs in (x) class ?".

"So just pick a class and let the model reference it when asnwering!" Bingo.

This is where we bring in the term positive class. The referenced class.

Let's say we pick cat as the positive class. This means the models predictions will answer the question "how sure are you that this new instance is a cat ?"

The model would return something like 0.9 if it thinks there is a cat. But if it thinks there is a dog(so it is like 5% sure there is a cat there) the returned value would be 0.05 (lower value because it doesn't think the sample belongs into the positive class)

>but if we flip it so dog(0) is the positive class then the model will predict 0.7 if it thinks it there a dog and 0.1 if it thinks there is a cat (because the positive class is dog)

The problem we run into with binary classification is that anything that isn't a cat is a dog. Or in more real terms everything that isn't a positive class is the negative class - even if it is neither. 
> This is where we turn this problem into a multi-class one, adding a third option/class.

First let's see what BCE does. This is the equation when **the positive class is 1**. 
> (we run into some small problems if the positive class is 0 ; ie. we are punishing the correct answers)

$$BCE(T_i, s_i)  =  - \frac{1}{n} *\sum_{i=1}^{n}(T_i * log(p_i) + (1 - T_i) * log(1 - p_i))$$

Loss for one example/sample is :

$$BCE(truth, prediction) = (loss\ for\ C_1\ + loss\ for\ C_2)$$

$loss\ for\ C_1 = T_i * log(s_i)$

$loss\ for\ C_2 = (1 - T_i) * log(1 - s_i)$

$T_i$ : truth ; binary value (either 1 or a 0)
> respresents the "class" of the example

$s_i$ : propability ; between 0 and 1 (closer it is to 1 the more the model thinks the sample belongs into the positive class)
> how confident is the model that the example belongs in the positive class

<br>

- when $T_i$ = 0 ; only loss for $C_2$ ; loss for $C_1$ falls off because we multiply it by 0

> $(1 - T_i)$ to $(1 - 0) = 1$, this changes nothing when truth is 0

$$1 * log(1 - s_i)$$ 

<br>

- when $y_i$ = 1 ; only loss for $C_1$ ; loss for $ C_2 $ falls of because we multiply it by 0, 
<br> $(1 - T_i) $ to $ (1 - 1) = 0$

> $y_i * something$ to $1 * something$ ; changes nothing 

$$1 * log(s_i)$$

We are left with logarthims of our models predictions.

Ideally the value sent to the log function should be high in both cases.
If the truth is 1 then the model should output a high value ($\hat{y}_i$ = 0.9+), but if the truth is 0 then the model should output a low value ($\hat{y}_i$ = 0.1-). But because the positive class is 1 what we send to log when truth is not the positive class(0) is $(1-0.1) = 0.9$ (a high value) - making the loss really small because the prediction is correct even if the models output isn't high.

#### **Categorical Cross-Entropy (CCE) - Softmax Loss**

> commonly used in multi-class classification tasks (1 or 2 or 3 .. ?)

When we have more than two classes or categories that our samples can have we use Categorical CE.

Here each example can belong to one class. The model predictions are withing vector(a 1D array), where each index presents a class and the value on that index the propability of the example belonging to that class.

> So let's expand on the binary dog/cat prediction model and make it multi-class by also being able to predict if there is a bird on an image. 
<br> Each category is associated with an index in a vector : dog = 0, cat = 1, bird = 2 so the array for EACH sample looks something like: [0.2, 0.1, 0.9]. (when testing the truths are also provided with a vector of the same size as the number of classes - so [0,0,1])
<br> The **max** value inside an array is the prediced class - in this case 0.9 at the index 2 - bird.

## (Mathetmatical) Optimization 

| |   |
|---|---|
| vector | quantity that has both magnitude(size) and direction(from point A to B) <br>; usually visualized as an arrow |
| vector field | space - where each point is a vector  |
| local minimum |   |
| iterative algorithm |   |
| multivariate function | function that takes in more than one argument/parameter  |
| |   |

> [! code and visualization here !](notebooks/02_useful_concepts.ipynb)

The looping trough data and adjusting paramters till the loss(result of a loss function) is minimized(or maximized) is called optimizing.

### Gradient descent

#### Gradient

It is a vector field. 

### Adam

## Training problems ?

### Vanishing gradients

### Exploding gradients

### Local Minima and Plateaus

# AI vs. machine learing vs. deep learning ?

>Once again time to clarify a lot of things because terms are often used interchangeably, making everything confusing
<br>

| | |
|---|---|
| field | broad topic/area of study (wide range of research and practice) |
| subfield  | a smaller, more specialized area of study withing a field (often emerge as a field grows/becomes more complex) |

Computer science is the **FIELD** here. A broad area of study spanning a VERY wide veriaiety of topics (ie. AI, networking, databases)

So Artificial intelligence is the **SUBFIELD** of computer science.
Because AI in itself spans through a very wide veriaiety of themes it's a **FIELD** in its own right.

Which then means machine learning (ML) is a **SUBFIELD** of AI, but it's complex enough to be a **FIELD** of its own. 
> (confused yet ?)

Deep learning (DL) is then a "sub-subfield" of AI, but also just a **SUBFIELD** of machine learning, it also utilizes other subfields in machine learning so we often refer to it as a "subset".

Many **SUBFIELDS** of AI (and even of machine learning) **use** machine learning methods. 
(they are still specialized areas of the study of artificial intelligence, they just aren't truly "standalone").

# Where is this all used ?

## Natural Language Processing (NLP)

## Computer vision (CV)

# ChatGPT
