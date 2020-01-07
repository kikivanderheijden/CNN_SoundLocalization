# custom metric

from tensorflow.keras import backend as K
import tensorflow as tf

def cos_distmet_2D_angular(y_true, y_pred):
    cos_sim = K.sum(y_true*y_pred, axis=1)/(K.sqrt(K.sum(K.square(y_true),axis=1))*K.sqrt(K.sum(K.square(y_pred),axis=1)))

    cosine_distance_degrees = tf.acos(K.clip(cos_sim,-1+K.epsilon(),1-K.epsilon()))/3.14159265359
        
    # take the mean across all samples because you have to return one scalar
    
    return cosine_distance_degrees