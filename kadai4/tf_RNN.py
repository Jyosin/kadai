import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 10 units.
model.add(layers.Dense(10))

model.summary()

model = keras.Sequential()
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
model.add(layers.GRU(256, return_sequences=True))

# The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
model.add(layers.SimpleRNN(128))

model.add(layers.Dense(10))

model.summary()

encoder_vocab = 1000
decoder_vocab = 2000

encoder_input = layers.Input(shape=(None,))
encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(
    encoder_input
)

# Return states in addition to output
output, state_h, state_c = layers.LSTM(64, return_state=True, name="encoder")(
    encoder_embedded
)
encoder_state = [state_h, state_c]

decoder_input = layers.Input(shape=(None,))
decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(
    decoder_input
)

# Pass the 2 states to a new LSTM layer, as initial state
decoder_output = layers.LSTM(64, name="decoder")(
    decoder_embedded, initial_state=encoder_state
)
output = layers.Dense(10)(decoder_output)

model = keras.Model([encoder_input, decoder_input], output)
model.summary()

batch_size = 64
# Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
# Each input sequence will be of size (28, 28) (height is treated like time).
input_dim = 28

units = 64
output_size = 10  # labels are from 0 to 9

# Build the RNN model
def build_model(allow_cudnn_kernel=True):
    # CuDNN is only available at the layer level, and not at the cell level.
    # This means `LSTM(units)` will use the CuDNN kernel,
    # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
    if allow_cudnn_kernel:
        # The LSTM layer with default options uses CuDNN.
        lstm_layer = keras.layers.LSTM(units, input_shape=(None, input_dim))
    else:
        # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
        lstm_layer = keras.layers.RNN(
            keras.layers.LSTMCell(units), input_shape=(None, input_dim)
        )
    model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_size),
        ]
    )
    return model

def getkmnist():

    x_test_data = np.load("/Users/wangruqin/VScode/kadai1/kadai2/KMNIST/kmnist-test-imgs.npz") 
    x_test = x_test_data['arr_0']
    
    y_test_data = np.load("/Users/wangruqin/VScode/kadai1/kadai2/KMNIST/kmnist-test-labels.npz") 
    y_test = y_test_data['arr_0']

    x_train_data = np.load("/Users/wangruqin/VScode/kadai1/kadai2/KMNIST/kmnist-train-imgs.npz") 
    x_train = x_train_data['arr_0']

    y_train_data = np.load("/Users/wangruqin/VScode/kadai1/kadai2/KMNIST/kmnist-train-labels.npz") 
    y_train = y_train_data['arr_0']

    return x_train,y_train,x_test,y_test

x_train, y_train, x_test, y_test = getkmnist()
x_train, x_test = x_train / 255.0, x_test / 255.0
sample, sample_label = x_train[0], y_train[0]

model = build_model(allow_cudnn_kernel=True)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)

train_loss = []
train_acc = []
test_loss = []
train_loss = []

history1 = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=25)
train_loss = history1.history['loss']
train_acc = history1.history['accuracy']
test_loss = history1.history['val_loss']
test_acc = history1.history['val_accuracy']
plt.plot(train_loss,label='train loss')
plt.plot(test_loss,label='test loss')
# plt.plot(train_acc,label='train_loss')
# plt.plot(test_acc,label='test_loss')

plt.legend(loc='best')
plt.grid()
plt.show()

plt.plot(train_acc,label='train_acc')
plt.plot(test_acc,label='test_acc')

plt.legend(loc='best')
plt.grid()
plt.show()



