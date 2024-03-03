


# 4. predmet ; izdelek : DEEP LEARNING - the basics


# What even is artificial intelligence ?

**Artificial intelligence** is a branch of computer science focused on developing and reaserching systems made to preform tasks that require human-like intelligence. <br>
(the term "AI" may also be used to describe such machines/software). 

AIs are generally quite specialized, often designed to preform only one task(or a small range of tasks). For example just recognizing emotions or just recognizing shapes. We call such specilized AIs **narrow or "weak" AIs.**

(opposite: general AIs ; able to learn, understand and adapt - learn from experience, adapt to new/different situations ... currently only a goal in the field of AI)

# Machine learning

> Machine learning is a subfield of AI, that achives the preforming of tasks requring human-like intelligence with algorithms, creating models to preform the tasks without directly programming them to do so. 

AI is all about making systems able to preform tasks that generally require human-like intelligence right ? <br>

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
>Methods withing machine learning are also just procedures, that tell the model what to do with the data.  <br> (they are algorithms - still part of a model, making them methods withing a method if you will)
>
>Let's draw a very clear line on what each thing is
>
>| |   |
>|---|---|
>| model | is the program that we give data to in order to preform the task that requires human-like intelligence. <br> (want to predict housing prices based on house size ? - you create a model to do it) |
>| method | procedures to process data withing a model <br> (they process the housing size-price data ; there can be many within a model) |


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
| regression |   |

|Model architectures||
|-|-|
| Regression models | 
| Decision tree |  |
| (Artificial) Neural network - (A)NN| model inspired by the biological neural network <br> - connections of neurons data travels through|
| |   |

## Methods

Methods are procedures for accomplishing or approaching something - generally a goal. <br> In machine learning methods are both parts of a training algorithm used to train a model, and parts of its structre. 

Learning paradigms generally tell us how a model interacts with data during training. Do we give it labeled data and tell it what things are ? Should the model discover patterns by itself ? And so on.

| Learning paradigms|  |
|-|-|
| Supervised learning      | training algorithm uses labled training data (tries getting closer to the correct label)                  |
| Unsupervised learning    | training algorithm uses unlabled training data - discoveres patterns by itself                               |
| Semi-Supervised learning | training algorithm uses both labled and unlabled data <br> (usually used when labled data is scarce/expensive)             |
| Reinforcement learning   | training algorithm trains by interacting - receiving feedback (rewards/penalties) 
| Self-supervised learning   | |
| |   |

There are types of methods that utlize learning paradigms, and do not fit very well into the categorization above. They generally tell us what else the model does with the data given - be it labled, unlabled or something in between.

|||
|-|-|
| Feature/representation learning      | model automatically identifies/extracts features from raw data <br> unsupervised, supervised)     
|Association rule learning| model identifies 'rules' to store - to process data with <br> IF data_is_something THEN output |
| |   |

### (Artificial) Neural Networks

Artificial neural networks are machine learning models, inspired by the biological neural network.

The layred structure comes in handy when we have large amounts of data that needs to be processed.

| |   |
|---|---|
| neuron/node/unit | cells, that recieve signal, process it and then forward it  |
| layer |  building block, groups up nodes that preform specific operations on data |
| weight | how important a value from a neuron has <br> (generally multiplied - so if a weight is really low the output value will also be low - having less impact)  |
| treshold| a point something has to cross |
| link |   |
| |   |

In the artificial neural networks neurons are also referred to as nodes or units. They take in data either from some external input data (ie. we give it) or from other nodes in the network, process it (run the given data through a function/method that returns a new value of the neuron) and produce an output - that is then sent to other nodes or treated as the final output.

Neural networks are models made up of nodes, that are connected together. These nodes are usually combined into layers.  Nodes of one layer connect only to nodes of the immediately preceding and immediately following layers.

| |   |
|---|---|
| input layer | layer that recieves external data  |
| output layer | layer that produces the final result  |
| hidden layer(s) | all layers (0+) between the input layer and the output layer of a neural network   |
| |   |

When a neural network has 2 or more layers, it falls into the category of DNNs - deep learning networks.

The nodes can connect to the following layers in differnt ways:

| |   |
|---|---|
| |   |



Nodes generally also have a weights, these values represent the "strenght" or importance of a connection between neurons in adjescent layers. 

>
>A node sends its ouput to a node (or many nodes) in the next layer through a connection/link. There exist one connection for each node it sends its output to.
(connection is a pair; current node - next node)
>
>The weight on the connection between the node sending its output and the node recieving it tell the recieving node how important the value is in calculating its own output.
>
>Each input value a node recives is multiplied by the weight of the connection it recieved it from, getting a "weighted input".
>
>All the weighted inputs are then summed up and sent to an activation function and this transformed input (output of a node) is then sent onward.


Node connections also have something called **bias term/unit/neuron**. It is a parameter associated with each node in an network - a treshold that the sum of weighted inputs has to reach/surpass in order for the node to "fire"(to send the summed input to an actiavtion function and forward the output).


The functions that data recived by a node is sent to are called **activation functions**

#### Activation functions



| |   |
|---|---|
| in-features (input) |   |
| out-features (output) |   |
| |   |





# Deep learning

## CNNs (Convolutional Neural Networks)




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


# How do we use any of this ?


# Where do we run into problems ?



# ChatGPT