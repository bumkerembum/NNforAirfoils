from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
x_len = y_Veri.shape[0]

#######DRAG##########
#Standartize(normalize) the data
scaler_y_drag = StandardScaler()
y_Train_drag_scaled = scaler_y_drag.fit_transform(y_Train[:, 0].reshape(-1, 1)).ravel()

regr.fit(X_Train,y_Train_drag_scaled)
y_p_drag_scaled = regr.predict(X_Veri)

# Predict and rescale the predictions back to the original scale
y_p_drag_scaled = regr.predict(X_Veri)
y_p_drag = scaler_y_drag.inverse_transform(y_p_drag_scaled.reshape(-1, 1)).ravel()

relative_drag = (np.absolute((y_Veri[:,0] - y_p_drag) / y_Veri[:,0])) * 100 

plt.figure(0)
plt.scatter(xx,relative_drag)
plt.ylim(0, 500)
plt.title("Network Drag, Y limiti 500")

#now remove values higher han the condition
threshold_drag = 5000
mask_drag = relative_drag <= threshold_drag

y_p_filtered_drag = y_p_drag[mask_drag]
y_Veri_filtered_drag = y_Veri[mask_drag]


#recalculate error for better variables & visualation
relative_filtered_drag = np.mean((np.absolute((y_Veri_filtered_drag[:,0] - y_p_filtered_drag) / y_Veri_filtered_drag[:,0])) * 100)
rmse_drag = np.sqrt(np.mean((y_Veri_filtered_drag[:,0] - y_p_filtered_drag)**2))
median_abs_drag = np.median(np.abs(y_Veri_filtered_drag[:,0] - y_p_filtered_drag))



#######LIFT##########
#Standartize(normalize) the data
scaler_y_lift = StandardScaler()
y_Train_lift_scaled = scaler_y_lift.fit_transform(y_Train[:, 1].reshape(-1, 1)).ravel()

regr.fit(X_Train,y_Train_lift_scaled)
y_p_lift_scaled = regr.predict(X_Veri)

# Predict and rescale the predictions back to the original scale
y_p_lift_scaled = regr.predict(X_Veri)
y_p_lift = scaler_y_lift.inverse_transform(y_p_lift_scaled.reshape(-1, 1)).ravel()


relative_lift = (np.absolute((y_Veri[:,1] - y_p_lift) / y_Veri[:,1])) * 100 

plt.figure(1)
plt.scatter(xx,relative_lift)
plt.ylim(0, 100)
plt.title("Network Lift, Y limiti 500")

#now remove values higher than the condition
threshold_lift = 100
mask_lift = relative_lift <= threshold_lift

y_p_filtered_lift = y_p_lift[mask_lift]
y_Veri_filtered_lift = y_Veri[mask_lift]


#recalculate error for better variables & visualation
relative_filtered_lift = np.mean((np.absolute((y_Veri_filtered_lift[:,1] - y_p_filtered_lift) / y_Veri_filtered_lift[:,1])) * 100)
rmse_lift = np.sqrt(np.mean((y_Veri_filtered_lift[:,1] - y_p_filtered_lift)**2))
median_abs_lift = np.median(np.abs(y_Veri_filtered_lift[:,1] - y_p_filtered_lift))



#######MOMENT##########
#Standartize(normalize) the data
scaler_y_moment = StandardScaler()
y_Train_moment_scaled = scaler_y_moment.fit_transform(y_Train[:, 2].reshape(-1, 1)).ravel()

regr.fit(X_Train,y_Train_moment_scaled)
y_p_moment_scaled = regr.predict(X_Veri)

# Predict and rescale the predictions back to the original scale
y_p_moment_scaled = regr.predict(X_Veri)
y_p_moment = scaler_y_moment.inverse_transform(y_p_moment_scaled.reshape(-1, 1)).ravel()


relative_moment = (np.absolute((y_Veri[:,2] - y_p_moment) / y_Veri[:,2])) * 100 

plt.figure(2)
plt.scatter(xx,relative_moment)
plt.ylim(0, 100)
plt.title("Network Moment, Y limiti 2000")

#now remove values higher than the condition
threshold_moment = 100
mask_moment = relative_moment <= threshold_moment

y_p_filtered_moment = y_p_moment[mask_moment]
y_Veri_filtered_moment = y_Veri[mask_moment]

aaa = np.absolute((y_Veri_filtered_moment[:,2] - y_p_filtered_moment) / y_Veri_filtered_moment[:,2]) * 100
#recalculate error for better variables & visualation
relative_filtered_moment = np.mean((np.absolute((y_Veri_filtered_moment[:,2] - y_p_filtered_moment) / y_Veri_filtered_moment[:,2])) * 100)
rmse_moment = np.sqrt(np.mean((y_Veri_filtered_moment[:,2] - y_p_filtered_moment)**2))
median_abs_moment = np.median(np.abs(y_Veri_filtered_moment[:,2] - y_p_filtered_moment))
