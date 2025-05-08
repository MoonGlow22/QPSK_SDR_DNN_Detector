import numpy as np
import tensorflow as tf
import scipy.io as sio

def bit_err(y_true, y_pred):
    err = 1 - tf.reduce_mean(
        tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.sign(y_pred - 0.5),
                    tf.cast(tf.sign(y_true - 0.5), tf.float32)
                ),
                tf.float32),
            1))
    return err

# Load the same data structure you used for training
append = '15__64'  # Make sure this matches your trained model
matlab = sio.loadmat('./MATLAB/sdr_data'+append+'.mat')

# Load test data and preprocess exactly like in training
Rx_test = matlab['Rx_test']
M_test = matlab['M_test']

# Parameters (must match training parameters)
block_size = 4
mu = 2
n_output = mu*(block_size-1)

# Preprocess test data exactly like in training
X_test = Rx_test.reshape((-1, 2))  # Reshape into [I,Q] pairs
num_test_samples = X_test.shape[0] // block_size
X_test = X_test[:num_test_samples*block_size].reshape((num_test_samples, 2*block_size))

Y_test = M_test.reshape((-1,))
num_Y_test_samples = Y_test.shape[0] // (mu*block_size)
Y_test = Y_test[:num_Y_test_samples*mu*block_size].reshape((num_Y_test_samples, mu*block_size))
Y_test = Y_test[:, 2:2+n_output]  # Remove pilot bits

# Recreate the model architecture exactly as during training
n_input = 2*block_size
n_hidden_1 = 100
n_hidden_2 = 50
n_hidden_3 = 20

inputs = tf.keras.Input(shape=(n_input,))
temp = tf.keras.layers.Dense(n_hidden_1, activation='relu')(inputs)
temp = tf.keras.layers.BatchNormalization()(temp)
temp = tf.keras.layers.Dense(n_hidden_2, activation='relu')(temp)
temp = tf.keras.layers.BatchNormalization()(temp)
temp = tf.keras.layers.Dense(n_hidden_3, activation='relu')(temp)
temp = tf.keras.layers.BatchNormalization()(temp)
outputs = tf.keras.layers.Dense(n_output, activation='sigmoid')(temp)
model = tf.keras.Model(inputs, outputs)

# Compile with the same metrics (though not strictly needed for prediction)
model.compile(optimizer='adam', loss='mse', metrics=[bit_err])

# Load the weights (choose either the validation or training weights)
weights_path = 'nnval.'+append+'.weights.h5'  # or 'nntrain.'+append+'.weights.h5'
model.load_weights(weights_path)

# Make predictions
predictions = model.predict(X_test)

# The predictions will be in the range [0,1] due to sigmoid activation
# You can threshold them at 0.5 to get binary predictions
binary_predictions = (predictions > 0.5).astype(int)

# Calculate bit error rate
bit_errors = np.sum(binary_predictions != Y_test)
total_bits = Y_test.size
ber = bit_errors / total_bits

#%% DNN evaluation

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_true_flat = Y_test.flatten()
y_pred_flat = binary_predictions.flatten()

# Confusion matrix hesapla
cm = confusion_matrix(y_true_flat, y_pred_flat)

# Görselleştirme (normalize edilmemiş)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.xlabel('Tahminler')
plt.ylabel('Gerçek Değerler')
plt.title('Confusion Matrix')
plt.show()

# Detaylı metrikler
TN, FP, FN, TP = cm.ravel()
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\nDetaylı Metrikler:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-Score: {f1_score:.4f}")
print(f"Bit Error Rate (BER): {1 - accuracy:.4f}")

#%% Conventional Method Evaluation (Constellation Decoder)
def conventional_qpsk_demodulator(signal):
    # QPSK demodulation by quadrant detection
    real_part = signal[:, 0]  # I component
    imag_part = signal[:, 1]  # Q component
    
    # Decision boundaries at 0 for both axes
    bits_i = (real_part > 0).astype(int)
    bits_q = (imag_part > 0).astype(int)
    
    # Interleave I and Q bits (QPSK to bits mapping)
    bits = np.empty(2 * len(signal), dtype=int)
    bits[0::2] = bits_i
    bits[1::2] = bits_q
    
    return bits

# Process test data for conventional demodulator
Rx_test_blocks = Rx_test.reshape((-1, 2))  # Reshape into [I,Q] pairs
conventional_bits = conventional_qpsk_demodulator(Rx_test_blocks)

# Truncate to match DNN output size (remove pilot bits)
conventional_bits = conventional_bits[:Y_test.size].reshape(Y_test.shape)

# Calculate conventional BER
conv_bit_errors = np.sum(conventional_bits != Y_test)
conv_ber = conv_bit_errors / Y_test.size

#%% Comparison of DNN vs Conventional Method
print("\n=== Performance Comparison ===")
print(f"DNN Bit Error Rate (BER): {ber:.4f}")
print(f"Conventional BER: {conv_ber:.4f}")
print(f"Improvement: {((conv_ber - ber)/conv_ber)*100:.2f}% reduction in BER")

# Plot BER comparison
plt.figure(figsize=(8, 5))
methods = ['Conventional', 'DNN']
bers = [conv_ber, ber]
colors = ['red', 'green']

bars = plt.bar(methods, bers, color=colors)
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER Comparison: Conventional vs DNN Detector')

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()

# Confusion matrix for conventional method
conv_cm = confusion_matrix(Y_test.flatten(), conventional_bits.flatten())

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
sns.heatmap(conv_cm, annot=True, fmt='d', cmap="Reds")
plt.xlabel('Predictions')
plt.ylabel('True Values')
plt.title('Conventional Method Confusion Matrix')

plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.xlabel('Predictions')
plt.ylabel('True Values')
plt.title('DNN Confusion Matrix')

plt.tight_layout()
plt.show()

