import os


import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
import matplotlib.pyplot as plt
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


def Eager_model():
    train_loss = []
    train_acc = []
    test_loss = []
    train_loss = []

    # mnist_images, mnist_labels,x_test,y_test = getkmnist()

    mnist_images, mnist_labels,x_test,y_test = getkmnist()
    mnist_images = tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32)
    mnist_labels = tf.cast(mnist_labels, tf.int64)

    dataset = tf.data.Dataset.from_tensor_slices((mnist_images,mnist_labels))
    dataset = dataset.shuffle(60000).batch(64)    

    test_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(x_test[...,tf.newaxis]/255, tf.float32),
    tf.cast(y_test,tf.int64)))
    test_dataset = test_dataset.shuffle(10000).batch(64)
    callbacks = [keras.callbacks.TensorBoard(log_dir="./logs")]



    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64,[3,3], activation='relu',
                            input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(128,[3,3], activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(256,[3,3], activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(0.001)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mnist_model.compile(loss = loss_object,
                        optimizer = optimizer,
                        metrics = ['accuracy'])
    mnist_model.summary()

    history1 = mnist_model.fit(dataset,validation_data = test_dataset,epochs = 25,callbacks=callbacks)
    # keras.utils.plot_model(mnist_model,'cnn_model.png',show_shapes = True)
    train_loss = history1.history['loss']
    train_acc = history1.history['accuracy']
    test_loss = history1.history['val_loss']
    test_acc = history1.history['val_accuracy']
    plt.plot(train_loss,label='train loss')
    plt.plot(test_loss,label='test loss')


    plt.legend(loc='best')
    plt.grid()
    plt.show()

    plt.plot(train_acc,label='train_acc')
    plt.plot(test_acc,label='test_acc')

    plt.legend(loc='best')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    Eager_model()

