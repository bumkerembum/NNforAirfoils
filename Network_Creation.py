import random
import numpy as np
import time
import pickle
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import pandas as pd

# Delete any previously loaded model if there are any
tf.compat.v1.reset_default_graph()

#Import Files
x_tr_data = pd.read_excel('file_location\file.xlsx', header = None)     # Training set's input
X_Train = np.array(x_tr_data)
                                                                                                                                        
x_veri_data = pd.read_excel('file_location\file.xlsx', header = None)   # Verification set's input
X_Veri = np.array(x_veri_data)

y_tr_data = pd.read_excel('file_location\file.xlsx', header = None)    # Training set's output
y_Train = np.array(y_tr_data)

y_veri_data = pd.read_excel('file_location\file.xlsx', header = None)  # Verification set's output
y_Veri = np.array(y_veri_data)


# Read dimensions of the data
nTrain = X_Train.shape[0]
nVeri = X_Veri.shape[0]
dim = X_Veri.shape[1]


# Set up NN parameters
INPUT_DIM = dim
OUTPUT_DIM = 3
NUM_SAMPLES =  int(X_Train.shape[0]/3)
NUM_TRAINING = X_Train.shape[0]
NUM_VALIDATING = X_Veri.shape[0]
NUM_HIDDEN = 58
learning_rate = 0.007

tstart = time.time()

class SobolevNetwork:
    def __init__(self, input_dim, num_hidden):
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.W1 = tf.Variable(tf.random.normal([self.input_dim, 49],stddev=0.1))
        self.b1 = tf.Variable(tf.ones([49]))
        self.W2 = tf.Variable(tf.random.normal([49, 49],stddev=0.1))
        self.b2 = tf.Variable(tf.ones([49]))
        self.W3 = tf.Variable(tf.random.normal([49, 49],stddev=0.1))
        self.b3 = tf.Variable(tf.ones([49]))
        self.W4 = tf.Variable(tf.random.normal([49, 49],stddev=0.1))
        self.b4 = tf.Variable(tf.ones([49]))
        self.W5 = tf.Variable(tf.random.normal([49, 49],stddev=0.1))
        self.b5 = tf.Variable(tf.ones([49]))
        self.W6 = tf.Variable(tf.random.normal([49, 3],stddev=0.1))
        self.b6 = tf.Variable(tf.ones([3]))
        self.weights = [(self.W1, self.b1), (self.W2, self.b2), (self.W3, self.b3), (self.W4, self.b4), (self.W5, self.b5),(self.W6, self.b6)]


    def forward(self, X):
        #Input layer
        out = X
        #Hidden layers
        W,b = self.weights[0]
        out = tf.nn.tanh(tf.matmul(out, W) + b)

        W,b = self.weights[1]
        out = tf.nn.tanh(tf.matmul(out, W) + b)

        W,b = self.weights[2]
        out = tf.nn.relu(tf.matmul(out, W) + b)

        W,b = self.weights[3]
        out = tf.nn.sigmoid(tf.matmul(out, W) + b)
        
        W,b = self.weights[4]
        out = tf.nn.relu(tf.matmul(out, W) + b)
        
        #Output layer
        W,b = self.weights[-1]
        out = tf.matmul(out, W) + b
        return out
    
# Define tensors and operations
X = tf.compat.v1.placeholder(tf.float32, shape=[None, INPUT_DIM],name='X')
y = tf.compat.v1.placeholder(tf.float32, shape=[None,OUTPUT_DIM],name='y')

model = SobolevNetwork(INPUT_DIM, NUM_HIDDEN)
y_p = model.forward(X)

predict_named = tf.compat.v1.identity(y_p, "prediction")

# Defining loss functions
loss_drag = tf.reduce_mean(tf.pow(y_p[:,0] - y[:,0], 2))
loss_lift = tf.reduce_mean(tf.pow(y_p[:,1] - y[:,1], 2))
loss_moment = tf.reduce_mean(tf.pow(y_p[:,2] - y[:,2], 2))
loss = (loss_drag + loss_lift + loss_moment) / 3

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)   # Define optimizer
train_op = optimizer.minimize(loss)

#Start  saving the model
saver = tf.compat.v1.train.Saver()


with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # Training loop
    num_epochs = 1000  # Define the number of epochs
    batch_size = 700   # Define the batch size on each iteration
    
    for epoch in range(num_epochs):
         
        for i in range(0, NUM_TRAINING, batch_size):
            
            batch_indices = np.random.choice(nTrain, batch_size)
            
            X_batch = X_Train[batch_indices]
            y_batch = y_Train[batch_indices]
            feed_dict = {X: X_batch, y: y_batch}
            _, train_loss, train_loss_drag, train_loss_lift, train_loss_moment = sess.run(
                [train_op, loss, loss_drag, loss_lift, loss_moment], feed_dict=feed_dict
            )

        # Print the training loss after each epoch
        print(f'Epoch {epoch + 1}, Training Loss: {train_loss}, Drag Loss: {train_loss_drag}, Lift Loss: {train_loss_lift}, Moment Loss: {train_loss_moment}')

    # Validation
    feed_dict = {X: X_Veri, y: y_Veri}
    validation_loss, validation_loss_drag, validation_loss_lift, validation_loss_moment = sess.run(
        [loss, loss_drag, loss_lift, loss_moment], feed_dict=feed_dict
    )
    print(f'Validation Loss: {validation_loss}, Drag Loss: {validation_loss_drag}, Lift Loss: {validation_loss_lift}, Moment Loss: {validation_loss_moment}')

    # Save the model
    saver.save(sess, 'file_location/model.ckpt')
    print("Model saved")



tfinal1 = time.time() - tstart
print('Final time: '+str(tfinal1))

    
