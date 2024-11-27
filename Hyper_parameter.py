import random
import numpy as np
import time
import pickle
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import pandas as pd
import optuna
from sklearn.preprocessing import StandardScaler

tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()


# Import Files
x_tr_data = pd.read_excel('//home//kerem//Desktop//Airfoil//scenario4//training_x.xlsx', header = None)
X_Train = np.array(x_tr_data)
                                                                                                                                        
x_veri_data = pd.read_excel('//home//kerem//Desktop//Airfoil//scenario4//validation_x.xlsx', header = None)
X_Veri = np.array(x_veri_data)

y_tr_data = pd.read_excel('//home//kerem//Desktop//Airfoil//scenario4//training_y.xlsx', header = None)
y_Train = np.array(y_tr_data)

y_veri_data = pd.read_excel('//home//kerem//Desktop//Airfoil//scenario4//validation_y.xlsx', header = None)
y_Veri = np.array(y_veri_data)


scaler_drag = StandardScaler()
scaler_lift = StandardScaler()
scaler_moment = StandardScaler()

y_Train_scaled = np.column_stack((
    scaler_drag.fit_transform(y_Train[:, 0].reshape(-1, 1)).ravel(),
    scaler_lift.fit_transform(y_Train[:, 1].reshape(-1, 1)).ravel(),
    scaler_moment.fit_transform(y_Train[:, 2].reshape(-1, 1)).ravel()
))

y_Veri_scaled = np.column_stack((
    scaler_drag.transform(y_Veri[:, 0].reshape(-1, 1)).ravel(),
    scaler_lift.transform(y_Veri[:, 1].reshape(-1, 1)).ravel(),
    scaler_moment.transform(y_Veri[:, 2].reshape(-1, 1)).ravel()
))


# Read dimensions of the data
nTrain = X_Train.shape[0]
nVeri = X_Veri.shape[0]
dim = X_Veri.shape[1]

tstart = time.time()

# Define the SN class
class SobolevNetwork:
    def __init__(self, input_dim, hidden_dims, activation_funcs):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation_funcs = activation_funcs
        self.weights = []
        prev_dim = input_dim
        for dim in hidden_dims:
            W = tf.Variable(tf.random.normal([prev_dim, dim], stddev=0.1))
            b = tf.Variable(tf.ones([dim]))
            self.weights.append((W, b))
            prev_dim = dim
        self.W_out = tf.Variable(tf.random.normal([prev_dim, 3], stddev=0.1))
        self.b_out = tf.Variable(tf.ones([3]))

    def forward(self, X):
        out = X
        for i, (W, b) in enumerate(self.weights):
            activation = self.activation_funcs[i]
            out = activation(tf.matmul(out, W) + b)
        out = tf.matmul(out, self.W_out) + self.b_out
        return out

# Hyperparameter optimization function
def objective(trial):
    
    num_epochs = 1000
    batch_size = 700
    
    # Hyperparameters to tune
    num_hidden_layers = trial.suggest_int('num_hidden_layers', low=1, high=5, step=1)
    hidden_units = trial.suggest_int('n_units', low=1, high=64, step=1)  # single value for all layers
    hidden_dims = [hidden_units] * num_hidden_layers
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    activation_functions = {
        'relu': tf.nn.relu,
        'sigmoid': tf.nn.sigmoid,
        'tanh': tf.nn.tanh,
        'leaky_relu': tf.nn.leaky_relu
    }
    hidden_activations = [trial.suggest_categorical(f'activation_l{i}', list(activation_functions.values())) for i in range(num_hidden_layers)]

    # Define the network
    model = SobolevNetwork(dim, hidden_dims, hidden_activations)
    
    X = tf.compat.v1.placeholder(tf.float32, shape=[None, dim], name='X')
    y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3], name='y')
    
    y_p = model.forward(X)

    
    loss_drag = tf.reduce_mean(tf.abs((y_p[:,0] - tf.reshape(y, [batch_size, 3])[:,0])/tf.reshape(y,[batch_size,3])[:,0]))
    loss_lift = tf.reduce_mean(tf.abs((y_p[:,1] - tf.reshape(y, [batch_size, 3])[:,1])/tf.reshape(y,[batch_size,3])[:,1]))
    loss_moment = tf.reduce_mean(tf.abs((y_p[:,2] - tf.reshape(y, [batch_size, 3])[:,2])/tf.reshape(y,[batch_size,3])[:,2]))
    loss = (loss_drag + loss_lift + loss_moment) / 3
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)
    
    # Training
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())    
    
    for epoch in range(num_epochs):
        batch_indices = np.random.choice(nTrain, batch_size)
        X_batch = X_Train[batch_indices]
        y_batch = y_Train_scaled[batch_indices]
        sess.run(train_op, feed_dict={X: X_batch, y: y_batch})
    
    # Validation
    y_valid_pred = sess.run(y_p, feed_dict={X: X_Veri})
    
    
    validation_loss_drag = np.mean(np.abs((y_valid_pred[:,0] - y_Veri_scaled[:,0])/y_Veri_scaled[:,0]))
    validation_loss_lift = np.mean(np.abs((y_valid_pred[:,1] - y_Veri_scaled[:,1])/y_Veri_scaled[:,1]))
    validation_loss_moment = np.mean(np.abs((y_valid_pred[:,2] - y_Veri_scaled[:,2])/y_Veri_scaled[:,2]))
  
    validation_loss = (validation_loss_drag + validation_loss_lift + validation_loss_moment) /3
     
    
    # Updata best loss
    
    #if validation_loss_drag <= objective.best_val_loss_drag and validation_loss_lift <= objective.best_val_loss_lift and validation_loss_moment <= objective.best_val_loss_moment:
        #objective.best_val_loss_drag = validation_loss_drag
        #objective.best_val_loss_lift = validation_loss_lift
        #objective.best_val_loss_moment = validation_loss_moment
        #objective.best_val_loss = validation_loss
    
    # Burada hata çıkabilir optizzzasyonun best validation seçme methodu değişik örneğin: 
    # 10 10 10 = 30
    #  8 10  7 = 25  benim method için ok
    # 15  7  3 = 25 benim method için ok değil ama optimizasyon için ok 
  
    if validation_loss <= objective.best_val_loss:
       objective.best_val_loss_drag = validation_loss_drag
       objective.best_val_loss_lift = validation_loss_lift
       objective.best_val_loss_moment = validation_loss_moment
       objective.best_val_loss = validation_loss
    
  
    
    print("\n")
    print(f"Trial {trial.number} - Loss Drag: {validation_loss_drag:.7f}, Loss Lift: {validation_loss_lift:.7f}, Loss Moment: {validation_loss_moment:.7f}")

    # Calculate criterion
    re = np.linalg.norm(y_valid_pred - y_Veri) / np.linalg.norm(y_Veri) * 100
    print(f"Trial {trial.number} - Criterion: {re:.4f}")
    
    if re < objective.best_re:
        objective.best_re = re
        objective.best_trial = trial.number

    print(f"Best criterion so far: {objective.best_re:.4f} in trial {objective.best_trial}")
    print("\n")
    print(f"Best Validation Losses: Drag={objective.best_val_loss_drag:.7f}, Lift={objective.best_val_loss_lift:.7f}, Moment={objective.best_val_loss_moment:.7f}")
    print(f"Best Total Loss: {objective.best_val_loss}")
    print("\n\n")

    return validation_loss

objective.best_val_loss = float("inf")
objective.best_val_loss_drag = float('inf')
objective.best_val_loss_lift = float('inf')
objective.best_val_loss_moment = float('inf')
objective.best_re = 1000   # initial value
objective.best_trial = 0   # initial value

# Run the hyperparameter optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500)

# Output the best hyperparameters
print("Best hyperparameters: ", study.best_params)
tfinal1 = (time.time() - tstart)/3600
print('Final time: '+str(tfinal1))
