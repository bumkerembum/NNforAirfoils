import random
import numpy as np
import time
import pickle
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

tf.compat.v1.reset_default_graph()


#Import Files
x_tr_data = pd.read_excel('location', header = None)
X_Train = np.array(x_tr_data)
                                                                                                                                        
x_veri_data = pd.read_excel('location', header = None)
X_Veri = np.array(x_veri_data)

y_tr_data = pd.read_excel('location', header = None)
y_Train = np.array(y_tr_data)

y_veri_data = pd.read_excel('location', header = None)
y_Veri = np.array(y_veri_data)


scaler_lift = StandardScaler()
scaler_moment = StandardScaler()

y_Train_scaled = np.column_stack((
    scaler_lift.fit_transform(y_Train[:, 1].reshape(-1, 1)).ravel(),
    scaler_moment.fit_transform(y_Train[:, 2].reshape(-1, 1)).ravel()
))

y_Veri_scaled = np.column_stack((
    scaler_lift.transform(y_Veri[:, 1].reshape(-1, 1)).ravel(),
    scaler_moment.transform(y_Veri[:, 2].reshape(-1, 1)).ravel()
))



# Read dimensions of the data
nTrain = X_Train.shape[0]
nVeri = X_Veri.shape[0]
dim = X_Veri.shape[1]


# Set up NN parameters
INPUT_DIM = dim
OUTPUT_DIM = 2
NUM_SAMPLES =  int(X_Train.shape[0]/3)
NUM_TRAINING = X_Train.shape[0]
NUM_VALIDATING = X_Veri.shape[0]
NUM_HIDDEN = 54
learning_rate = 0.0056
tstart = time.time()

class Network:
    def __init__(self, input_dim, num_hidden):
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.W1 = tf.Variable(tf.random.normal([self.input_dim, NUM_HIDDEN],stddev=0.1))
        self.b1 = tf.Variable(tf.ones([NUM_HIDDEN]))
        self.W2 = tf.Variable(tf.random.normal([NUM_HIDDEN, NUM_HIDDEN],stddev=0.1))
        self.b2 = tf.Variable(tf.ones([NUM_HIDDEN]))
        self.W3 = tf.Variable(tf.random.normal([NUM_HIDDEN, NUM_HIDDEN],stddev=0.1))
        self.b3 = tf.Variable(tf.ones([NUM_HIDDEN]))
        self.W4 = tf.Variable(tf.random.normal([NUM_HIDDEN, NUM_HIDDEN],stddev=0.1))
        self.b4 = tf.Variable(tf.ones([NUM_HIDDEN]))
        self.W5 = tf.Variable(tf.random.normal([NUM_HIDDEN, 2],stddev=0.1))
        self.b5 = tf.Variable(tf.ones([2]))
        self.weights = [(self.W1, self.b1), (self.W2, self.b2), (self.W3, self.b3), (self.W4, self.b4), (self.W5, self.b5)]


    def forward(self, X):
        #Input layer
        out = X
        #Hidden layers
        W,b = self.weights[0]
        out = tf.nn.tanh(tf.matmul(out, W) + b)
        
        W,b = self.weights[1]
        out = tf.nn.relu(tf.matmul(out, W) + b)
        
        W,b = self.weights[2]
        out = tf.nn.leaky_relu(tf.matmul(out, W) + b)
        
        W,b = self.weights[3]
        out = tf.nn.sigmoid(tf.matmul(out, W) + b)

        #Output layer
        W,b = self.weights[-1]
        out = tf.matmul(out, W) + b
        return out
    
# Define tensors and operations
X = tf.compat.v1.placeholder(tf.float32, shape=[None, INPUT_DIM],name='X')
y = tf.compat.v1.placeholder(tf.float32, shape=[None,OUTPUT_DIM],name='y')

model = Network(INPUT_DIM, NUM_HIDDEN)
y_p = model.forward(X) 

predict_named = tf.compat.v1.identity(y_p, "prediction")

loss_lift = tf.reduce_mean(tf.abs((y_p[:,0] - y[:,0])/y[:,0]))
loss_moment = tf.reduce_mean(tf.abs((y_p[:,1] - y[:,1])/y[:,1]))
loss = (loss_lift + loss_moment) / 2

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)   
train_op = optimizer.minimize(loss)

#Start  saving the model
saver = tf.compat.v1.train.Saver()


with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # Training loop
    num_epochs = 1000  # Define the number of epochs
    batch_size = 700
    
    
    plot_loss = []
    
    for epoch in range(num_epochs):
        indices = np.random.permutation(X_Train.shape[0])       
        for i in range(0, NUM_TRAINING, batch_size):
            
            batch_indices = indices[i:i + batch_size]
            
            X_batch = X_Train[batch_indices]
            y_batch = y_Train_scaled[batch_indices]
            feed_dict = {X: X_batch, y: y_batch}
            _, train_loss, train_loss_lift, train_loss_moment = sess.run(
                [train_op, loss, loss_lift, loss_moment], feed_dict=feed_dict
            )

        # Print the training loss after each epoch
        print(f'Epoch {epoch + 1}, Training Loss: {train_loss}')


        plot_loss.append(train_loss)


    # Validation
    feed_dict = {X: X_Veri, y: y_Veri_scaled}
    validation_loss, validation_loss_lift, validation_loss_moment = sess.run(
        [loss, loss_lift, loss_moment], feed_dict=feed_dict
    )
    print(f'Validation Loss: {validation_loss}')




tfinal1 = time.time() - tstart
print('Final time: '+str(tfinal1))

plot_x = list(range(1, num_epochs+1))
plt.plot(plot_x, plot_loss)
plt.show()

    
