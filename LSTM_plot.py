#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:36:48 2018

@author: hujingyi
"""


# we actually use the ml to predic data, and the news,investors make educated guesses by analyzing data will read company hoistory and industry trends 
#there are so many data points that go into making a prediction the prices.
#but we know that the prices are totally random and unpredictable
# why did theses top firms like Morgan family and Citigroup higher quantitives analyst to build predictive models?
#we know that many expert quietly sitting in front of computers screens, in fact about 70% of all borders on Wall Street are now placed by software,
#we're now living in the age of the algorithme.

from keras.layers.core import Dense, Activation, Dropout 
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm, time #helper libraries
 
#we are going to build a deep learning model to predict stock prices records of prices.
#in finance, the filed of quantative analysts about 25 years old, even now it's still not fully accepted, understood or widely used.
#it's cetainly teh study of how certain variables corralate with stock price behavior.
# in the past few years, we've seen a lot of academic paper published using neural nets to predict stock prices with varying degrees of success 
# and now with libraries like Tensorflow we can build powerful predictive models trained on massive data sets.
# so let's build our own model using care off with a tensorflow backend.
#Keras 可以基于两个Backend，一个是 Theano，一个是 Tensorflow。
#如果选择 Tensorflow 的话呢，Keras 就使用 Tensorflow 在底层搭建神经网络。

# for  our traing data wll be using the daily closing price of the SMP 500 from jan 2002 to August 2016
# this a seires of data points indexed in time order or a time series.
# our goal will be to predict the closing price for any given data after training

# =============================================================================
# step 1 Load data
# =============================================================================
X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', 50, True)

# we can load are they using a custom load data function assentially just read our CSV file into an array of values 
# and normalizes them, normalize them improve convergences
# we use this equation to normalize each value to reflect percentage changes from the starting point
# so we'll divide each price by the initial price and substract one
# when our model later makes prediction will denormalize the data with p0(ni+1) to get a real world number out of it

# =============================================================================
# step2 Build Model
# =============================================================================

# to build our model will first initialize it and sequentialize it and it will be a linear stack of layers 
# then we will add our first layer which is a LTSM layer wo 
# to what is it ?????
# it's easy to recall the word for word could we sing them backwards? no
 # the reason for this is we learn these words in a sequence it's a conditional memory
 # we can access the word if we act of the word for it
 # but feed forwrd neural nets don't have the capacity to remember , because what will happen depends on every fram ebefore it
 # we will need a way to allow information to persist and that's why we will use a recurrent neural nets 
 # it can accept sequences as vectors as inputs so recall that for feed-formard neural nets the hidden 
 # layers wights are based only on the input data(input->hidden->output) but in the rnn the hiden layer is a combination of the input data 
 # at the current time step and the hidden the hidden layer in a previous step(input + prev_hidden)-> hidden->output
 # the hiden layer is constantly changing as it gets more inputs and the only way to reach these hidden 
 # states is with the correct sequence of inputs. 
 # this is how the memory is incorporated in and we can model this process mathematically so this hidden 
 # they are capable to connect previous data with the current.
 # Hoshua NGO called it the vanishing gradient problem in one of his most frequently cited papers piled learning long-term 
 # dependencies with gradient descent is difficult.
 # a popular solution to this is a modification to recurring that's called long short tem memory.
 # Normallym neurons are unit that apply an activation function like a sigmoid to a linear combination of thier
 # inputs
 # in an LTM recurrent net we 
 
 #Step 2 Build Model
model = Sequential()

# so for LSTM layer we will set our dimension to 1 and say we want 50 units in this layer
# setting return sequences to true means this layers output is always set into the next layer
# all its activations can be seen as a sequence of predictions the first layer has made from the input 
# sequence will add 20 percent dropout to this layer
model.add(LSTM(
    input_dim=1,
    output_dim=50,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))


# then initialize our second layer as another lstm with 100 units and set our return sequence to fall on
# it since its output is noly fed to the next layer at the end of this sequence 
# it doesn't help put a prediction for the sequence instead a prediction vector 
# for the whole sequence will use the linear dense layer to aggregate the data
# from the precdiction vector into on single value
# than we can compile our model using a popular loss function called mean squared error and use gradient decent
# as our optimizer labeled "rmsprop"
model.add(Dense(
    output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print ('compilation time : ', time.time() - start)



# =============================================================================
# #Step 3 Train the model
# =============================================================================
model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=1,
    validation_split=0.05)


# =============================================================================
# #Step 4 - Plot the predictions!
# =============================================================================
predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
lstm.plot_results_multiple(predictions, y_test, 50)
# we will train our model with the function then we can test it to see what it predicts 
# for the next 50 steps at several points in our graph 
# and visualize it using that with mapplotlit
# it seems that for a lot of the price movement espacially the big ones
# there is quite a correlation between our models prediction and the actual data 
# so it's time to some money!!

# but will our model be able to correcly predict the closing price one hundred percent of time?
# the answer is no
# it's an analystical tool to help us make educated guesses about the direction of the market,
# that's slightly better than random
# so to break it down, conclusions:
# 1. rnn can model sequential data because the hidden state is affected by the input and 
# the previous hidden state 

# 2. a solution to the vanishing gradient is to use LSTM(long short time memory) cells to remember long-term
# dependencies
# 3. we can use  LSTM networks to make decisions for time series data easily using Keras and Tensorflow


###############################################################################
#Keras is an open source neural network library written in Python. 
#It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit, Theano, or MXNet.
#Designed to enable fast experimentation with deep neural networks, 
#it focuses on being user-friendly, modular, and extensible.


#我们得到误差, 而且在 反向传递 得到的误差的时候, 他在每一步都会 乘以一个自己的参数 W. 
#如果这个 W 是一个小于1 的数, 比如0.9. 这个0.9 不断乘以误差, 误差传到初始时间点也会是一个接近于零的数, 
#所以对于初始时刻, 误差相当于就消失了. 我们把这个问题叫做梯度消失或者梯度弥散 Gradient vanishing. 
#反之如果 W 是一个大于1 的数, 比如1.1 不断累乘, 则到最后变成了无穷大的数, RNN被这无穷大的数撑死了, 
#这种情况我们叫做剃度爆炸, Gradient exploding. 这就是普通 RNN 没有办法回忆起久远记忆的原因.

