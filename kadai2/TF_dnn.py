import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np


def mnistset():

    mnist = tf.keras.datasets.mnist
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train,x_test = x_train/255.0, x_test/255.0
    print(np.shape(x_train))
    print(np.shape(y_train))
    return x_train,y_train,x_test,y_test

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
    x_train,y_train,x_test,y_test = getkmnist()

    callbacks = [keras.callbacks.TensorBoard(log_dir="./logs")]
    dataset = tf.data.Dataset.from_tensor_slices(((x_train,y_train,x_test,y_test)))
    dataset = dataset.batch(32)




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
    x = My_Dense(784,units=1024)(x)
    x = keras.activations.relu(x)

    x = My_Dense(1024,units=1024)(x)
    x = keras.activations.relu(x)

    x = My_Dense(1024,units=1024)(x)
    x = keras.activations.relu(x)
        
    x = My_Dense(1024,units=1024)(x)
    x = keras.activations.relu(x)

    x = My_Dense(1024,units=10)(x)
    Outputs = keras.activations.softmax(x)

    model = keras.Model(Inputs,Outputs)
    model.summary()
    keras.utils.plot_model(model,'my_fc_model.png',show_shapes = True)
    return model
def My_Train(): 
    model = My_model()
    train_loss = []
    train_acc = []
    test_loss = []
    train_loss = []
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(0.001),
                  metrics = ['accuracy'])
    x_train,y_train,x_test,y_test = getkmnist()
    callbacks = [keras.callbacks.TensorBoard(log_dir="./logs")]

    dataset = tf.data.Dataset.from_tensor_slices(((x_train,y_train)))
    dataset = dataset.shuffle(60000).batch(32)

    test_dataset = tf.data.Dataset.from_tensor_slices(((x_test,y_test)))
    test_dataset = test_dataset.shuffle(10000).batch(32)
    # model.fit(dataset,epochs = 10,callbacks=callbacks)
    history1 = model.fit(dataset,validation_data = test_dataset,epochs = 25,callbacks=callbacks)
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

    

if __name__ == "__main__":
    My_Train()
    # getkmnist()