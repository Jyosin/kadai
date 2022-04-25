import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np


def mnistset():

    mnist = tf.keras.datasets.mnist
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train,x_test = x_train/255.0, x_test/255.0
    print(np.shape(x_train))
    print(np.shape(y_train))
    return x_train,y_train,x_test,y_test

def FCmodel():

    Inputs = keras.Input(shape = (28,28,1),name = 'FC_input')
# //这里必须要在形状的最后声明层数为1
    x = layers.Flatten()(Inputs)
# //展开变成向量
    x = layers.Dense(units=64,activation='relu',name='D1')(x)
    x = layers.Dense(units=32,activation='relu',name='D2')(x)
    Outputs = layers.Dense(units=10,activation='softmax',name='outputs')(x)

    model = keras.Model(Inputs,Outputs)
    model.summary()
# //绘制总结表格
    keras.utils.plot_model(model,'fc_model.png',show_shapes = True)
    return model

def Train():
    model = FCmodel()
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(0.01),
                  metrics = ['accuracy'])
    x_train,y_train,_,_ = mnistset()

    callbacks = [keras.callbacks.TensorBoard(log_dir="./logs")]
# //回调
    dataset = tf.data.Dataset.from_tensor_slices(((x_train,y_train)))
    dataset = dataset.batch(32)
    model.fit(dataset,epochs = 100,callbacks=callbacks)


class My_Dense(layers.Layer):
    def __init__(self,Input_dim,units):
        super(My_Dense,self).__init__()
        self.units = units

        self.w = self.add_weight(shape = (Input_dim,units),
                                 initializer = 'random_normal',
                                 trainable = True)
        # //必须告知维度，声明可被训练
        self.b = self.add_weight(shape = (units,),
                                 initializer='zeros',
                                 trainable = True)
    def call(self,inputs):
        return tf.matmul(inputs,self.w) + self.b
def My_model():
    Inputs = keras.Input(shape = (28,28),name = 'FC_input')
    x = layers.Flatten()(Inputs)
    x = My_Dense(784,units=64)(x)

    x = keras.activations.relu(x)
    x = My_Dense(64,units=32)(x)
    x = keras.activations.relu(x)
    x = My_Dense(32,units=10)(x)
    Outputs = keras.activations.softmax(x)

    model = keras.Model(Inputs,Outputs)
    model.summary()
    keras.utils.plot_model(model,'my_fc_model.png',show_shapes = True)
    return model
def My_Train(): 
    model = My_model()
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(0.01),
                  metrics = ['accuracy'])
    x_train,y_train,_,_ = mnistset()
    callbacks = [keras.callbacks.TensorBoard(log_dir="./logs")]

    dataset = tf.data.Dataset.from_tensor_slices(((x_train,y_train)))
    dataset = dataset.batch(32)
    model.fit(dataset,epochs = 10,callbacks=callbacks)

if __name__ == "__main__":
    My_Train()