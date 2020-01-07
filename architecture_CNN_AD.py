#------------------------------------------------------------------------------
# Specifications
#------------------------------------------------------------------------------
# specify directories
dir_wrfiles = "/home/jovyan/DNN_ESANN_dev" # for testing on DSRI
#dir_wrfiles = r"C:\Users\kiki.vanderheijden\Documents\PostDoc_Auditory\DeepLearning" # for testing locally

# import libraries
from tensorflow.keras import layers
from tensorflow.keras import models # contains different types of models (use sequential model here?)
from tensorflow.keras import optimizers # contains different types of back propagation algorithms to train the model, 
                                        # including sgd (stochastic gradient
#from CustLoss_MSE import cust_mean_squared_error # note that in this loss function, the axis of the MSE is set to 1
from CustLoss_cosine_distance_angular import cos_dist_2D_angular
from CustMet_cosine_distance_angular import cos_distmet_2D_angular

# specify parameters
modelname   = 'model19'
time_sound  = 2000 # input dimension 1 (time)
nfreqs      = 99 # input dimension 2 (frequencies)

#------------------------------------------------------------------------------
# Define model architecture
#------------------------------------------------------------------------------
# left channel
in1                 = layers.Input(shape=(time_sound,nfreqs,1)) # define input (rows, columns, channels (only one in my case))
model_l_conv1       = layers.Conv2D(16,(1,3),activation='relu', padding = 'same')(in1) # define first layer and input to the layer
model_l_conv1_mp    = layers.MaxPooling2D(pool_size = (1,2))(model_l_conv1)
model_l_conv1_mp_do = layers.Dropout(0.2)(model_l_conv1_mp)

# right channel
in2                 = layers.Input(shape=(time_sound,nfreqs,1)) # define input
model_r_conv1       = layers.Conv2D(16,(1,3),activation='relu', padding = 'same')(in2) # define first layer and input to the layer
model_r_conv1_mp    = layers.MaxPooling2D(pool_size = (1,2))(model_r_conv1)
model_r_conv1_mp_do = layers.Dropout(0.2)(model_r_conv1_mp)

# merged
model_final_merge       = layers.Concatenate(axis = -1)([model_l_conv1_mp_do, model_r_conv1_mp_do]) 
model_final_conv1       = layers.Conv2D(32,(3,3),activation='relu', padding = 'same')(model_final_merge)
model_final_conv1_mp    = layers.MaxPooling2D(pool_size = (2,2))(model_final_conv1)
model_final_conv1_mp_do = layers.Dropout(0.2)(model_final_conv1_mp)

model_final_flatten = layers.Flatten()(model_final_conv1_mp_do)
model_final_dropout = layers.Dropout(0.2)(model_final_flatten) # dropout for regularization
predicted_coords    = layers.Dense(2, activation = 'tanh')(model_final_dropout) # I have used the tanh activation because our outputs should be between -1 and 1

#------------------------------------------------------------------------------
# Create model
#------------------------------------------------------------------------------
# create
model = models.Model(inputs = [in1,in2], outputs = predicted_coords) # create
# compile
model.compile(loss = cos_dist_2D_angular, optimizer = optimizers.Adam(), metrics=['cosine_proximity','mse',cos_distmet_2D_angular])
# print summary
model.summary()
# save
model.save(dir_wrfiles+'/DNN_'+modelname+'.h5') # save model

