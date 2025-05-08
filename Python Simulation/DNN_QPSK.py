# pylint: disable=W0614
import numpy as np
import tensorflow as tf
import sys
import scipy.io as sio

def bit_err(y_true, y_pred):
    err = 1 - tf.reduce_mean(
        tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.sign(
                        y_pred - 0.5),
                    tf.cast(
                        tf.sign(
                            y_true - 0.5),
                        tf.float32)
                ),
                tf.float32),
            1))
    return err

## load data
append = '15__64'
matlab = sio.loadmat(
    './MATLAB/sdr_data'+append+'.mat')
M = matlab['M']
MRx = matlab['MRx']
Rx = matlab['Rx']
M_val = matlab['M_val']
MRx_val = matlab['MRx_val']
Rx_val = matlab['Rx_val']
M_test = matlab['M_test']
MRx_test = matlab['MRx_test']
Rx_test = matlab['Rx_test']

## initialize parameters
start = matlab['start'][0, 0]
delay = matlab['delay'][0, 0]  # 58//2
epoch_size = matlab['epoch_size'][0, 0]  # 4*100000
batch_size = 128
val_size = matlab['val_size'][0, 0]  # 4*10000
test_size = matlab['test_size'][0, 0]
sps = 8                     #samples per symbol
mu = 2
block_size = 4              #frame size

# nn layer sizes
n_input = 2*block_size
n_hidden_1 = 100
n_hidden_2 = 50
n_hidden_3 = 20
n_output = mu*(block_size-1)

# Fix: Check the shape of Rx first and reshape accordingly
print("Original Rx shape:", Rx.shape)

# Fix: Reshape Rx (using the correct dimensions based on our data generation)
# In our data generator, Rx is already flattened as [I, Q, I, Q, ...]
# So we need to reshape it to have block_size pairs of I/Q values
X = Rx.reshape((-1, 2))  # First reshape into [I,Q] pairs
num_samples = X.shape[0] // block_size
X = X[:num_samples*block_size].reshape((num_samples, 2*block_size))

# Similarly for validation and test sets
X_val = Rx_val.reshape((-1, 2))
num_val_samples = X_val.shape[0] // block_size
X_val = X_val[:num_val_samples*block_size].reshape((num_val_samples, n_input))

X_test = Rx_test.reshape((-1, 2))
num_test_samples = X_test.shape[0] // block_size
X_test = X_test[:num_test_samples*block_size].reshape((num_test_samples, n_input))

# Reshape M (ensuring we get block_size pairs of bits)
Y = M.reshape((-1,))
num_Y_samples = Y.shape[0] // (mu*block_size)
Y = Y[:num_Y_samples*mu*block_size].reshape((num_Y_samples, mu*block_size))
Y = Y[:, 2:2+n_output]              #remove pilot bits

Y_val = M_val.reshape((-1,))
num_Y_val_samples = Y_val.shape[0] // (mu*block_size)
Y_val = Y_val[:num_Y_val_samples*mu*block_size].reshape((num_Y_val_samples, mu*block_size))
Y_val = Y_val[:, 2:2+n_output]

Y_test = M_test.reshape((-1,))
num_Y_test_samples = Y_test.shape[0] // (mu*block_size)
Y_test = Y_test[:num_Y_test_samples*mu*block_size].reshape((num_Y_test_samples, mu*block_size))
Y_test = Y_test[:, 2:2+n_output]

# Print some shapes to verify
print("Input X shape:", X.shape)
print("Output Y shape:", Y.shape)

## NN
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=epoch_size//batch_size*5,
    decay_rate=0.5,
    staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
inputs = tf.keras.Input(shape=(n_input,))
temp = tf.keras.layers.BatchNormalization()(inputs)
temp = tf.keras.layers.Dense(n_hidden_1, activation='relu')(inputs)
temp = tf.keras.layers.BatchNormalization()(temp)
temp = tf.keras.layers.Dense(n_hidden_2, activation='relu')(temp)
temp = tf.keras.layers.BatchNormalization()(temp)
temp = tf.keras.layers.Dense(n_hidden_3, activation='relu')(temp)
temp = tf.keras.layers.BatchNormalization()(temp)
outputs = tf.keras.layers.Dense(n_output, activation='sigmoid')(temp)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=optimizer, loss='mse', metrics=[bit_err])
model.summary()
print(append)
#model.load_weights(filepath=filename)
#model.load_weights(filepath=path+'/templastsave.h5')
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='nnval.'+append+'.weights.h5', monitor='val_bit_err',
                                                verbose=0, save_best_only=True, mode='min', save_weights_only=True)
saver = tf.keras.callbacks.LambdaCallback(
    on_batch_begin=lambda batch, logs: tf.keras.Model.save_weights(model, filepath='templasave.h5'))
history = model.fit(
    x=X, y=Y,
    #steps_per_epoch=500,
    epochs=100,
    validation_data=(X_val, Y_val),
    callbacks=[checkpoint],
    verbose=1,
)
tf.keras.Model.save_weights(model, filepath='nntrain.'+append+'.weights.h5')
print('fully trained')
y2 = model.evaluate(
    x=X_test, y=Y_test
)
print('best val')
model.load_weights(filepath='nnval.'+append+'.weights.h5')
y1 = model.evaluate(
    x=X_test, y=Y_test
)
dict_mat = {
    'ber_nntrain': y2,
    'ber_nnval': y1,
}
sio.savemat(
    'NN'+append+'.mat',
    dict_mat
)
print(append)