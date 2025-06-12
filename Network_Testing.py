import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

tf.compat.v1.reset_default_graph()

# Import Files
test_x_data = pd.read_excel('location', header=None)
Test_x = np.array(test_x_data)
                                                                                                                                        
test_y_data = pd.read_excel('location', header=None)
Test_y = np.array(test_y_data)


#Get scaling data 

y_tr_data = pd.read_excel('location', header = None)
y_Train = np.array(y_tr_data)


scaler_lift = StandardScaler()
scaler_moment = StandardScaler()

y_Train_scaled = np.column_stack((
    scaler_lift.fit_transform(y_Train[:, 1].reshape(-1, 1)).ravel(),
    scaler_moment.fit_transform(y_Train[:, 2].reshape(-1, 1)).ravel(),

))

y_Test_scaled = np.column_stack((
    scaler_lift.transform(Test_y[:, 1].reshape(-1, 1)).ravel(),
    scaler_moment.transform(Test_y[:, 2].reshape(-1, 1)).ravel(),
))


# Start a session
with tf.compat.v1.Session() as sess:
    # Load the meta graph and weights
    saver = tf.compat.v1.train.import_meta_graph('location/model.ckpt.meta', clear_devices=True)
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint('location/'))

    # Get the default graph
    graph = tf.compat.v1.get_default_graph()
    
    # Get the placeholders and operation by name
    X = graph.get_tensor_by_name("X:0")
    prediction = graph.get_tensor_by_name("prediction:0")

    # Prepare the feed dictionary
    feed_dict = {X: Test_x}
    
    # Run the session to get the predictions
    Test_prediction_scaled = sess.run(prediction, feed_dict=feed_dict)
    
    # Print the results
    #print(f'Test Predictions: {Test_prediction}')


# Invert scaled Predictions

Test_prediction = np.column_stack((
    scaler_lift.inverse_transform(Test_prediction_scaled[:, 0].reshape(-1, 1)).ravel(),
    scaler_moment.inverse_transform(Test_prediction_scaled[:, 1].reshape(-1, 1)).ravel() 
    ))


# Calculations of error

lift_1 = np.mean(np.power(Test_prediction[:,0] - Test_y[:,1], 2))
moment_1 = np.mean(np.power(Test_prediction[:,1] - Test_y[:,2], 2))
loss_1 = (lift_1 + moment_1 ) / 2

print(f'For Root Mean Square Eror: \nLift: {lift_1}, Moment: {moment_1}, Total: {loss_1} \n\n')


lift_2 = (np.absolute((Test_y[:,1] - Test_prediction[:,0]) / Test_y[:,1])) * 100 
moment_2 = (np.absolute((Test_y[:,2] - Test_prediction[:,1]) / Test_y[:,2])) * 100 
loss_2 = (lift_2 + moment_2 ) / 2
xx = np.arange(200)
plt.figure(1)
plt.scatter(xx,lift_2)
#plt.ylim(0, 35)
plt.title("Network lift, Y limit 35") 
plt.figure(2)
plt.scatter(xx,moment_2)
#plt.ylim(0, 200)
plt.title("Network momen, Y limit 200") 
plt.figure(3)





#print(f'For Absolute Percentage Eror: \nDrag: {drag_2}, Lift: {lift_2},  Total: {loss_2} \n\n')

drag_error = []
lift_error = []
moment_error = []


for i in range(len(xx)):
    if moment_2[i] > 35:
        moment_error.append(i)
    
    if lift_2[i] > 35:
        lift_error.append(i)



common_items = set(drag_error).intersection(lift_error, moment_error)
