from tensorflow.keras import backend as K

def cust_mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=1)