# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 09:34:48 2022

@author: clevine1
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import seaborn as sb

#%%
#Data load in and preprocessing into scaled windows
data_FD001 = np.load('CMAPPS_FD001_nonsc.npz')
data_FD004 = np.load('CMAPPS_FD004_nonsc.npz')

x_train1 = data_FD001['x_train']
y_train1 = data_FD001['y_train']
x_train4 = data_FD004['x_train']
y_train4 = data_FD004['y_train']
x_test1 = data_FD001['x_test']
y_test1 = data_FD001['y_test']
x_test4 = data_FD004['x_test']
y_test4 = data_FD004['y_test']

scaler = MinMaxScaler(feature_range=(0,1))
x_train1 = scaler.fit_transform(x_train1)
x_test1 = scaler.transform(x_test1)
x_train4 = scaler.fit_transform(x_train4)
x_test4 = scaler.transform(x_test4)

x_train1 = x_train1.reshape(-1,30,14)
x_train4 = x_train4.reshape(-1,19,14)
x_test1 = x_test1.reshape(-1,30,14)
x_test4 = x_test4.reshape(-1,19,14)

#initialize rmse arrays (length 5 instead for the simple RNN runs, and for the final best model runs)
rmse_train1 = np.zeros(3)
rmse_test1 = np.zeros(3)
rmse_train4 = []
rmse_test4 = []

#initialize blocks of arrays where the rmse values will go upon testing

rmse_arr128_1= np.zeros([5,5])
rmse_arr256_1= np.zeros([5,5])
rmse_arr512_1= np.zeros([5,5])
rmse_arr128_2= np.zeros([5,5])
rmse_arr256_2= np.zeros([5,5])
rmse_arr512_2= np.zeros([5,5])
rmse_arr128_3= np.zeros([5,5])
rmse_arr256_3= np.zeros([5,5])
rmse_arr512_3= np.zeros([5,5])
rmse_gru128_1= np.zeros([4,4])
rmse_gru256_1= np.zeros([4,4])
rmse_gru128_2= np.zeros([4,4])
rmse_gru256_2= np.zeros([4,4])
rmse_lstm128_1= np.zeros([4,4])
rmse_lstm256_1= np.zeros([4,4])
rmse_lstm512_1= np.zeros([4,4])
rmse_lstm128_2= np.zeros([4,4])
rmse_lstm256_2= np.zeros([4,4])
rmse_lstm512_2= np.zeros([4,4])

#%%
#This contains two for-loops: the outer controls values for the number of neurons and the inner loop contains the learning rate values
#shown here is the two-layer LSTM model, the other lines were uncommented to run other conditions
#simpleRNN condition also had 64 neuron and 0.008 learning rate conditions, not included here
count=-1
count2=-1
for unit in [4,8,16,32]:
  count2 = count2+1
  count=-1
  for learning_rate in [0.0005,0.001,0.002,0.004]:
    count = count+1
    for i in range(3):

        #Clear the session.
        tf.keras.backend.clear_session()

        # Initialize the RNN and fit it.
        #Comment out lines of code that are not necessary. This was used to run the last model, which was 2-layer GRU batchsize 256.
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(x_train1.shape[1],x_train1.shape[2])),# Shape [batch, time, features] => [batch, time, lstm_units]
            #tf.keras.layers.SimpleRNN(units, return_sequences=True),
            #tf.keras.layers.SimpleRNN(units, return_sequences=True),
            #tf.keras.layers.SimpleRNN(units, return_sequences=True),
            #tf.keras.layers.GRU(32, return_sequences=True),
            tf.keras.layers.LSTM(units=unit, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            #tf.keras.layers.GRU(32, return_sequences=True),
            tf.keras.layers.LSTM(units=unit, return_sequences=True),
            #tf.keras.layers.Dense(units=unit, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=1)
        ])
        #Compile using the Adam optimizer.
        model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss=tf.keras.losses.MeanSquaredError(),
                    )
        model_history = model.fit(x_train1, 
                                  y_train1, 
                                  batch_size = 256, 
                                  epochs = 20, 
                                  validation_split = 0.15, 
                                  verbose=0)
        y_pred_train1 = model.predict(x_train1).reshape(-1,1)
        y_pred_test1 = model.predict(x_test1).reshape(-1,1)

        rmse_train1[i] = np.sqrt(np.mean(np.square(y_pred_train1 - y_train1.reshape(-1,1))))
        rmse_test1[i] = np.sqrt(np.mean(np.square(y_pred_test1 - y_test1.reshape(-1,1))))

        print('Train RMSE run number ',i+1,', neurons=',unit,', learning rate=', learning_rate,': ',rmse_train1[i])
        print('Test RMSE run number ',i+1,', neurons=',unit,', learning rate=', learning_rate,': ',rmse_test1[i])
    rmseavg = np.mean(rmse_test1)
    #averages three runs and sends it to a 4x4 array
    rmse_lstm256_2[count, count2] = rmseavg
#%%
#concatenate rows and then columns for rnn, gru, and lstm architectures
full_rnn_1 = np.concatenate((np.concatenate((rmse_arr128_1, rmse_arr256_1),axis=1),rmse_arr512_1),axis=1)
full_rnn_2 = np.concatenate((np.concatenate((rmse_arr128_2, rmse_arr256_2),axis=1),rmse_arr512_2),axis=1)
full_rnn_3 = np.concatenate((np.concatenate((rmse_arr128_3, rmse_arr256_3),axis=1),rmse_arr512_3),axis=1)
full_rnn = np.concatenate((np.concatenate((full_rnn_1, full_rnn_2),axis=0),full_rnn_3),axis=0)
print(full_rnn)
#%%
full_gru_1 = np.concatenate((rmse_gru128_1, rmse_gru256_1),axis=1)
full_gru_2 = np.concatenate((rmse_gru128_2, rmse_gru256_2),axis=1)
full_gru = np.concatenate((full_gru_1, full_gru_2),axis=0)
print(full_gru)
#%%
full_lstm_1 = np.concatenate((rmse_lstm128_1, rmse_lstm256_1),axis=1)
full_lstm_2 = np.concatenate((rmse_lstm128_2, rmse_lstm256_2),axis=1)
full_lstm = np.concatenate((full_lstm_1, full_lstm_2),axis=0)
print(full_lstm)
#%%
#Plot heatmaps in reversed colormap to visualize where the best RMSE parameters are
sb.heatmap(full_rnn, cmap='viridis_r')
sb.heatmap(full_gru, cmap='viridis_r')
sb.heatmap(full_lstm, cmap='viridis_r')
#%%
#the best performing was a gru with 32 neurons, batchsize 128, and learning rate 0.0005
#run for 70 epochs with two extra dense layers added
tf.keras.backend.clear_session()

        # Initialize the RNN and fit it.
model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(x_train1.shape[1],x_train1.shape[2])),# Shape [batch, time, features] => [batch, time, lstm_units]
            #tf.keras.layers.SimpleRNN(units, return_sequences=True),
            tf.keras.layers.GRU(32, return_sequences=True),
            #tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(32, return_sequences=True),
            #tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=4),
            tf.keras.layers.Dense(units=4),
            tf.keras.layers.Dense(units=1)
        ])
        #Compile using the Adam optimizer.
model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0005),
                loss=tf.keras.losses.MeanSquaredError(),
                    )
model_history = model.fit(x_train1, 
                                  y_train1, 
                                  batch_size = 128, 
                                  epochs = 70, 
                                  validation_split = 0.15, 
                                  verbose=2)
y_pred_train1 = model.predict(x_train1).reshape(-1,1)
y_pred_test1 = model.predict(x_test1).reshape(-1,1)
#%%
#Plot loss metrics
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('GRU 2-layer model, 2 Dense layers loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#show rmse metrics for train and test data
y_pred_train1 = model.predict(x_train1).reshape(-1,1)
y_pred_test1 = model.predict(x_test1).reshape(-1,1)

rmse_train1 = np.sqrt(np.mean(np.square(y_pred_train1 - y_train1.reshape(-1,1))))
rmse_test1 = np.sqrt(np.mean(np.square(y_pred_test1 - y_test1.reshape(-1,1))))

print('Train RMSE:',rmse_train1)
print('Test RMSE:',rmse_test1)
#%%
#show prediction plots for train and test data
plt.figure(figsize=(7,7))
plt.scatter(y_train1,y_pred_train1)
plt.title("Training Data Predictions")
plt.plot([0,1.01*y_test1.max()],[0,1.1*y_test1.max()], 'r')
plt.show()
plt.figure(figsize=(7,7))
plt.scatter(y_test1,y_pred_test1)
plt.title("Test Data Predictions")
plt.plot([0,1.01*y_test1.max()],[0,1.1*y_test1.max()], 'r')
plt.show()
#%%
#apply this model architecture to dataset FD004
tf.keras.backend.clear_session()

        # Initialize the RNN and fit it.
model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(x_train4.shape[1],x_train4.shape[2])),# Shape [batch, time, features] => [batch, time, lstm_units]
            #tf.keras.layers.SimpleRNN(units, return_sequences=True),
            tf.keras.layers.GRU(32, return_sequences=True),
            #tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(32, return_sequences=True),
            #tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=4),
            tf.keras.layers.Dense(units=4),
            tf.keras.layers.Dense(units=1)
        ])
        #Compile using the Adam optimizer.
model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0005),
                loss=tf.keras.losses.MeanSquaredError(),
                    )
model_history = model.fit(x_train4, 
                                  y_train4, 
                                  batch_size = 128, 
                                  epochs = 70, 
                                  validation_split = 0.15, 
                                  verbose=2)
#%%
#Plot loss metrics for FD004
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('GRU 2-layer model, 2 Dense layers loss: FD004')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#show rmse metrics for train and test data for FD004
y_pred_train4 = model.predict(x_train4).reshape(-1,1)
y_pred_test4 = model.predict(x_test4).reshape(-1,1)

rmse_train4 = np.sqrt(np.mean(np.square(y_pred_train4 - y_train4.reshape(-1,1))))
rmse_test4 = np.sqrt(np.mean(np.square(y_pred_test4 - y_test4.reshape(-1,1))))

print('Train RMSE:',rmse_train4)
print('Test RMSE:',rmse_test4)
#%%
#show prediction plots for train and test data for FD004
plt.figure(figsize=(7,7))
plt.scatter(y_train4,y_pred_train4)
plt.title("Training Data Predictions: FD004")
plt.plot([0,1.01*y_test4.max()],[0,1.1*y_test4.max()], 'r')
plt.show()
plt.figure(figsize=(7,7))
plt.scatter(y_test4,y_pred_test4)
plt.title("Test Data Predictions: FD004")
plt.plot([0,1.01*y_test4.max()],[0,1.1*y_test4.max()], 'r')
plt.show()
#%%
#Applying convolutional neural net in combination with simple RNN
#one for loop, which tests three different kernel sizes
#uses the best hyperparameters for 2-layer net as determined above
rmseavg = np.zeros(3)
counter=-1
for kernel in [10, 12, 14]:
  for i in range(3):
        #preprocessing of FD001 with convolutional neural network then applying RNN
        tf.keras.backend.clear_session()
        
        # Initialize the CNN-RNN and fit it.
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(x_train1.shape[1],x_train1.shape[2])),# Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.Reshape((x_train1.shape[1],x_train1.shape[2],1)),
            tf.keras.layers.Conv2D(16, (1,kernel), activation='relu'), #(feature maps, kernel size, activation function)
            tf.keras.layers.Reshape((30,(16*(15-kernel)))),
            tf.keras.layers.SimpleRNN(64, return_sequences=True),
            tf.keras.layers.SimpleRNN(64, return_sequences=True),
            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(units=1),
        ])
                #Compile using the Adam optimizer.
        model.compile(
                        optimizer=tf.keras.optimizers.Adam(0.001),
                        loss=tf.keras.losses.MeanSquaredError(),
                            )
        model_history = model.fit(x_train1, 
                                          y_train1, 
                                          batch_size = 128, 
                                          epochs = 30, 
                                          validation_split = 0.15, 
                                          verbose=1)
        y_pred_train1 = model.predict(x_train1).reshape(-1,1)
        y_pred_test1 = model.predict(x_test1).reshape(-1,1)
    
        rmse_train1[i] = np.sqrt(np.mean(np.square(y_pred_train1 - y_train1.reshape(-1,1))))
        rmse_test1[i] = np.sqrt(np.mean(np.square(y_pred_test1 - y_test1.reshape(-1,1))))
    
        print('Train RMSE run number ',i+1,', kernelsize = ',kernel, ': ',rmse_train1[i])
        print('Test RMSE run number ',i+1,', kernelsize = ',kernel, ': ',rmse_test1[i])
  counter=counter+1
  rmseavg[counter] = np.mean(rmse_test1)
#%%
#same as above, but with LSTM
#uses the best hyperparameters for 2-layer simple rnn as determined above
rmseavg = np.zeros(3)
counter=-1
for kernel in [10, 12, 14]:
  for i in range(3):
        #preprocessing of FD001 with convolutional neural network then applying RNN
        tf.keras.backend.clear_session()
        
        # Initialize the CNN-RNN and fit it.
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(x_train1.shape[1],x_train1.shape[2])),# Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.Reshape((x_train1.shape[1],x_train1.shape[2],1)),
            tf.keras.layers.Conv2D(16, (1,kernel), activation='relu'), #(feature maps, kernel size, activation function)
            tf.keras.layers.Reshape((30,(16*(15-kernel)))),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(units=1),
        ])
                #Compile using the Adam optimizer.
        model.compile(
                        optimizer=tf.keras.optimizers.Adam(0.001),
                        loss=tf.keras.losses.MeanSquaredError(),
                            )
        model_history = model.fit(x_train1, 
                                          y_train1, 
                                          batch_size = 128, 
                                          epochs = 30, 
                                          validation_split = 0.15, 
                                          verbose=1)
        y_pred_train1 = model.predict(x_train1).reshape(-1,1)
        y_pred_test1 = model.predict(x_test1).reshape(-1,1)
    
        rmse_train1[i] = np.sqrt(np.mean(np.square(y_pred_train1 - y_train1.reshape(-1,1))))
        rmse_test1[i] = np.sqrt(np.mean(np.square(y_pred_test1 - y_test1.reshape(-1,1))))
    
        print('Train RMSE run number ',i+1,', kernelsize = ',kernel, ': ',rmse_train1[i])
        print('Test RMSE run number ',i+1,', kernelsize = ',kernel, ': ',rmse_test1[i])
  counter=counter+1
  rmseavg[counter] = np.mean(rmse_test1)
#%%
#Run best CNN-Simple RNN network architecture (kernel width 10) for 70 epochs, five runs total and average
rmse_train1 = np.zeros(5)
rmse_test1 = np.zeros(5)
for i in range(5):
        #preprocessing of FD001 with convolutional neural network then applying RNN
        tf.keras.backend.clear_session()
        
        # Initialize the CNN-RNN and fit it.
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(x_train1.shape[1],x_train1.shape[2])),# Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.Reshape((x_train1.shape[1],x_train1.shape[2],1)),
            tf.keras.layers.Conv2D(16, (1,10), activation='relu'), #(feature maps, kernel size, activation function)
            tf.keras.layers.Reshape((30,(16*(15-10)))),
            tf.keras.layers.SimpleRNN(64, return_sequences=True),
            tf.keras.layers.SimpleRNN(64, return_sequences=True),
            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(units=1),
        ])
                #Compile using the Adam optimizer.
        model.compile(
                        optimizer=tf.keras.optimizers.Adam(0.001),
                        loss=tf.keras.losses.MeanSquaredError(),
                            )
        model_history = model.fit(x_train1, 
                                          y_train1, 
                                          batch_size = 128, 
                                          epochs = 70, 
                                          validation_split = 0.15, 
                                          verbose=1)
        y_pred_train1 = model.predict(x_train1).reshape(-1,1)
        y_pred_test1 = model.predict(x_test1).reshape(-1,1)
    
        rmse_train1[i] = np.sqrt(np.mean(np.square(y_pred_train1 - y_train1.reshape(-1,1))))
        rmse_test1[i] = np.sqrt(np.mean(np.square(y_pred_test1 - y_test1.reshape(-1,1))))
    
        print('Train RMSE run number ',i+1,': ',rmse_train1[i])
        print('Test RMSE run number ',i+1,': ',rmse_test1[i])
rmseavg = np.mean(rmse_test1)
print("average test RMSE: ",rmseavg)
#%%
#Run best CNN-LSTM architecture (kernel width of 10) for 50 epochs and then average RMSE of 5 runs
#uses best 2-layer LSTM architecture as determined above (not best simple rnn architecture)
rmse_train1 = np.zeros(5)
rmse_test1 = np.zeros(5)
for i in range(5):
        #preprocessing of FD001 with convolutional neural network then applying RNN
        tf.keras.backend.clear_session()
        
        # Initialize the CNN-RNN and fit it.
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(x_train1.shape[1],x_train1.shape[2])),# Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.Reshape((x_train1.shape[1],x_train1.shape[2],1)),
            tf.keras.layers.Conv2D(16, (1,10), activation='relu'), #(feature maps, kernel size, activation function)
            tf.keras.layers.Reshape((30,(16*(15-10)))),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(units=1),
        ])
                #Compile using the Adam optimizer.
        model.compile(
                        optimizer=tf.keras.optimizers.Adam(0.002),
                        loss=tf.keras.losses.MeanSquaredError(),
                            )
        model_history = model.fit(x_train1, 
                                          y_train1, 
                                          batch_size = 128, 
                                          epochs = 50, 
                                          validation_split = 0.15, 
                                          verbose=1)
        y_pred_train1 = model.predict(x_train1).reshape(-1,1)
        y_pred_test1 = model.predict(x_test1).reshape(-1,1)
    
        rmse_train1[i] = np.sqrt(np.mean(np.square(y_pred_train1 - y_train1.reshape(-1,1))))
        rmse_test1[i] = np.sqrt(np.mean(np.square(y_pred_test1 - y_test1.reshape(-1,1))))
    
        print('Train RMSE run number ',i+1,': ',rmse_train1[i])
        print('Test RMSE run number ',i+1,': ',rmse_test1[i])
rmseavg = np.mean(rmse_test1)
print("average test RMSE: ",rmseavg)
#%%
#Plot loss metrics
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('CNN-LSTM Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
#%%
#show prediction plots for train and test data
plt.figure(figsize=(7,7))
plt.scatter(y_train1,y_pred_train1)
plt.title("Training Data Predictions, CNN-LSTM")
plt.plot([0,1.01*y_test1.max()],[0,1.1*y_test1.max()], 'r')
plt.show()
plt.figure(figsize=(7,7))
plt.scatter(y_test1,y_pred_test1)
plt.title("Test Data Predictions, CNN-LSTM")
plt.plot([0,1.01*y_test1.max()],[0,1.1*y_test1.max()], 'r')
plt.show()