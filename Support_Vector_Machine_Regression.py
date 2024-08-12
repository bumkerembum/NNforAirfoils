from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Inputs of training data sat
x_tr_data = pd.read_excel(r'file_location\file.xlsx', header = None)
X_Train = np.array(x_tr_data)
# Inputs of test data
x_veri_data = pd.read_excel(r'file_location\file.xlsx', header = None)
X_Veri = np.array(x_veri_data)
# Output of training data
y_tr_data = pd.read_excel(r'file_location\file.xlsx', header = None)
y_Train = np.array(y_tr_data)
# Expected output of test dat
y_veri_data = pd.read_excel(r'file_location\file.xlsx', header = None)
y_Veri = np.array(y_veri_data)


regr = svm.SVR()
regr.fit(X_Train,y_Train[:,2])  # If using matrix instead of vector, specify the target column
y_p = regr.predict(X_Veri)

# Calculation percentage error
error = (np.absolute((y_Veri[:,2] - y_p) / y_Veri[:,2])) * 100 

x_len = y_Veri.shape[0]
xx = np.arange(x_len)
plt.scatter(xx,error)
plt.ylim(0, 500)
plt.title("Network Drag, Y limiti 150")
