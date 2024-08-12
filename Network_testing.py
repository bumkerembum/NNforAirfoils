import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Delete any previous loaded network models
tf.compat.v1.reset_default_graph()

# Import Files
test_x_data = pd.read_excel('file_location\file.xlsx', header=None)
Test_x = np.array(test_x_data)
                                                                                                                                        
test_y_data = pd.read_excel('file_location\file.xlsx', header=None)
Test_y = np.array(test_y_data)

# Start a session
with tf.compat.v1.Session() as sess:
    # Load the meta graph and weights
    saver = tf.compat.v1.train.import_meta_graph('file_location/your_model.meta', clear_devices=True)
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint('file_location/'))
    
    # Get the default graph
    graph = tf.compat.v1.get_default_graph()
    
    # Get the placeholders and operation by name
    X = graph.get_tensor_by_name("X:0")
    prediction = graph.get_tensor_by_name("prediction:0")  # Adjust this tensor name to match the actual output tensor from your model

    # Prepare the feed dictionary
    feed_dict = {X: Test_x}
    
    # Run the session to get the predictions
    Test_prediction = sess.run(prediction, feed_dict=feed_dict)
    
    # Print the results
    #print(f'Test Predictions: {Test_prediction}')

# Calculations of error

drag_1 = np.mean(np.power(Test_prediction[:,0] - Test_y[:,0], 2))
lift_1 = np.mean(np.power(Test_prediction[:,1] - Test_y[:,1], 2))
moment_1 = np.mean(np.power(Test_prediction[:,2] - Test_y[:,2], 2)) 
loss_1 = (drag_1 + lift_1 + moment_1) / 3

print(f'For Root Mean Square Eror: \nDrag: {drag_1}, Lift: {lift_1}, Moment: {moment_1}, Total: {loss_1} \n\n')

# Calculation percentage error
drag_2 = (np.absolute((Test_y[:,0] - Test_prediction[:,0]) / Test_y[:,0])) * 100 
lift_2 = (np.absolute((Test_y[:,1] - Test_prediction[:,1]) / Test_y[:,1])) * 100 
moment_2 = (np.absolute((Test_y[:,2] - Test_prediction[:,2]) / Test_y[:,2])) * 100 
loss_2 = (drag_2 + lift_2 + moment_2) / 3

x_len = Test_y[0]
xx = np.arange(x_len)
#plt.scatter(xx,lift_2)
#plt.scatter(xx,drag_2)
plt.scatter(xx,moment_2)

#plt.ylim(0, 150)
plt.title("Network Moment")

#print(f'For Absolute Percentage Eror: \nDrag: {drag_2}, Lift: {lift_2}, Moment: {moment_2}, Total: {loss_2} \n\n')







