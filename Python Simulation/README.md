Python code for simulating the experiment process

data_creator.py creates a random bit sequence, QPSK modulate it, add 19dB awgn, demodulate and save it in appropriate format. 

DNN_QPSK.py trains the model.

prediction_and_evaluation.py makes predictions both with DNN and Conventional Method (Constellation Decoder) and compares them.

nntrain.15__64.weights.h5 is the best model weights according to train data

nnval.15__64.weights.h5 is the best model weights according to validation data

There is the train, validation and test data in MATLAB folder in one piece
