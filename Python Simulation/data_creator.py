import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def generate_qpsk_data(num_frames=10000, block_size=4, sps=8, snr_db=15):
    """
    Generate QPSK modulated data with noise for neural network training
    
    Args:
        num_frames: Number of frames to generate
        block_size: Size of each frame in symbols
        sps: Samples per symbol
        snr_db: Signal-to-noise ratio in dB
    
    Returns:
        dict: Dictionary containing all required data matrices
    """
    mu = 2  # bits per symbol for QPSK
    pilot_bits = np.array([1, 1])  # pilot bits at the start of each frame
    
    # Generate random binary data (0s and 1s)
    M = np.random.randint(0, 2, size=(num_frames, mu*(block_size-1)))
    
    # Create frames with pilot bits
    frames = np.zeros((num_frames, mu*block_size))
    frames[:, 0:2] = pilot_bits  # Add pilot bits
    frames[:, 2:] = M  # Add data bits
    
    # QPSK Modulation
    # Map bit pairs to QPSK symbols
    # 00 -> -1-1j, 01 -> -1+1j, 10 -> 1-1j, 11 -> 1+1j
    modulated = np.zeros((num_frames, block_size), dtype=complex)
    
    for i in range(num_frames):
        for j in range(block_size):
            bit_pair = frames[i, j*2:(j+1)*2]
            if np.array_equal(bit_pair, [0, 0]):
                modulated[i, j] = -1 - 1j
            elif np.array_equal(bit_pair, [0, 1]):
                modulated[i, j] = -1 + 1j
            elif np.array_equal(bit_pair, [1, 0]):
                modulated[i, j] = 1 - 1j
            elif np.array_equal(bit_pair, [1, 1]):
                modulated[i, j] = 1 + 1j
    
    # Upsample to get samples per symbol
    upsampled = np.zeros((num_frames, block_size*sps), dtype=complex)
    for i in range(num_frames):
        for j in range(block_size):
            upsampled[i, j*sps] = modulated[i, j]
    
    # Apply pulse shaping (Root Raised Cosine)
    beta = 0.35  # roll-off factor
    span = 8  # filter span in symbols
    
    # Create RRC filter
    t = np.arange(-span*sps/2, span*sps/2 + 1)
    rrc_filter = np.zeros_like(t, dtype=float)
    
    # RRC filter formula
    for i, ti in enumerate(t):
        if abs(ti) == 0:
            rrc_filter[i] = 1.0 - beta + (4*beta/np.pi)
        elif abs(abs(ti) - sps/(4*beta)) < 1e-10:
            rrc_filter[i] = (beta/np.sqrt(2)) * ((1 + 2/np.pi) * np.sin(np.pi/(4*beta)) + 
                                               (1 - 2/np.pi) * np.cos(np.pi/(4*beta)))
        else:
            num = np.sin(np.pi*ti/sps*(1-beta)) + 4*beta*ti/sps*np.cos(np.pi*ti/sps*(1+beta))
            den = np.pi*ti/sps * (1 - (4*beta*ti/sps)**2)
            rrc_filter[i] = num / den
    
    # Normalize filter
    rrc_filter = rrc_filter / np.sqrt(np.sum(rrc_filter**2))
    
    # Apply filter to get transmitted signal
    transmitted = np.zeros((num_frames, block_size*sps + len(rrc_filter) - 1), dtype=complex)
    for i in range(num_frames):
        transmitted[i, :] = np.convolve(upsampled[i, :], rrc_filter)
    
    # Calculate signal power
    signal_power = np.mean(np.abs(transmitted)**2)
    
    # Calculate noise power based on SNR
    noise_power = signal_power / (10**(snr_db/10))
    
    # Add complex AWGN noise
    noise = np.sqrt(noise_power/2) * (np.random.randn(*transmitted.shape) + 
                                    1j * np.random.randn(*transmitted.shape))
    received = transmitted + noise
    
    # Apply matched filter (RRC) at the receiver
    filtered = np.zeros_like(received)
    for i in range(num_frames):
        filtered[i, :] = np.convolve(received[i, :], rrc_filter, 'same')
    
    # Sample at symbol rate
    center = (len(rrc_filter)-1)//2
    end = center + block_size*sps
    
    # Get the received symbols (complex)
    rx_complex = np.zeros((num_frames, block_size), dtype=complex)
    for i in range(num_frames):
        for j in range(block_size):
            idx = center + j*sps
            if idx < filtered.shape[1]:
                rx_complex[i, j] = filtered[i, idx]
    
    # Convert to real-valued representation for the neural network
    rx_real = np.zeros((num_frames*block_size, 2))
    for i in range(num_frames):
        for j in range(block_size):
            rx_real[i*block_size+j, 0] = rx_complex[i, j].real
            rx_real[i*block_size+j, 1] = rx_complex[i, j].imag
    
    # Reshape M for output
    M_out = np.zeros((num_frames*mu*block_size))
    for i in range(num_frames):
        M_out[i*mu*block_size:(i+1)*mu*block_size] = frames[i, :]
    
    # Reshape Rx for output (flattened version)
    Rx = rx_real.flatten()
    
    # Create MATLAB compatible version
    MRx = np.zeros_like(M_out)  # This would normally contain the demodulated bits
    
    # Split the data into training, validation, and test sets
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    train_size = int(num_frames * train_ratio)
    val_size = int(num_frames * val_ratio)
    test_size = int(num_frames * test_ratio)
    
    # Training data
    M_train = M_out[:train_size*mu*block_size]
    Rx_train = Rx[:train_size*block_size*2]
    MRx_train = MRx[:train_size*mu*block_size]
    
    # Validation data
    M_val = M_out[train_size*mu*block_size:(train_size+val_size)*mu*block_size]
    Rx_val = Rx[train_size*block_size*2:(train_size+val_size)*block_size*2]
    MRx_val = MRx[train_size*mu*block_size:(train_size+val_size)*mu*block_size]
    
    # Test data
    M_test = M_out[(train_size+val_size)*mu*block_size:]
    Rx_test = Rx[(train_size+val_size)*block_size*2:]
    MRx_test = MRx[(train_size+val_size)*mu*block_size:]
    
    # Create a dictionary to save as .mat file
    data_dict = {
        'M': M_train,
        'MRx': MRx_train,
        'Rx': Rx_train,
        'M_val': M_val,
        'MRx_val': MRx_val,
        'Rx_val': Rx_val,
        'M_test': M_test,
        'MRx_test': MRx_test,
        'Rx_test': Rx_test,
        'start': np.array([[0]]),
        'delay': np.array([[0]]),
        'epoch_size': np.array([[train_size]]),
        'val_size': np.array([[val_size]]),
        'test_size': np.array([[test_size]])
    }
    
    return data_dict

def save_qpsk_data(filename='sdr_data15__64.mat', num_frames=100000, block_size=4, sps=8, snr_db=15):
    """
    Generate QPSK data and save to a .mat file
    
    Args:
        filename: Name of the output .mat file
        num_frames: Number of frames to generate
        block_size: Size of each frame in symbols
        sps: Samples per symbol
        snr_db: Signal-to-noise ratio in dB
    """
    data_dict = generate_qpsk_data(num_frames, block_size, sps, snr_db)
    sio.savemat(filename, data_dict)
    print(f"Data saved to {filename}")
    
    # Print some statistics
    print(f"Total frames: {num_frames}")
    print(f"Training frames: {data_dict['epoch_size'][0][0]}")
    print(f"Validation frames: {data_dict['val_size'][0][0]}")
    print(f"Test frames: {data_dict['test_size'][0][0]}")
    print(f"SNR: {snr_db} dB")
    
    return data_dict

# Function to visualize generated QPSK constellation
def plot_constellation(data_dict, max_points=1000):
    """
    Plot the QPSK constellation from the generated data
    
    Args:
        data_dict: Dictionary containing the generated data
        max_points: Maximum number of points to plot
    """
    # Reshape Rx data to get I/Q pairs
    rx_data = data_dict['Rx'].reshape(-1, 2)
    
    if rx_data.shape[0] > max_points:
        indices = np.random.choice(rx_data.shape[0], max_points, replace=False)
        rx_data = rx_data[indices]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(rx_data[:, 0], rx_data[:, 1], alpha=0.6, marker='.')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(alpha=0.3)
    plt.title('QPSK Constellation Diagram')
    plt.xlabel('In-Phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()

# Usage example
if __name__ == "__main__":
    # Generate data with default parameters
    data_dict = save_qpsk_data(filename='./MATLAB/sdr_data15__64.mat', 
                              num_frames=100000, 
                              block_size=4, 
                              sps=8, 
                              snr_db=19)
    
    # Optionally plot the constellation
    plot_constellation(data_dict)