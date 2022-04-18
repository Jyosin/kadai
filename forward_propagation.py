from cgi import test
import numpy as np
class Network():
    def __init__(self):
        self.nn_architecture = [
            {"input_dim": 2, "output_dim": 4, "activation": "relu"},
            {"input_dim": 4, "output_dim": 6, "activation": "relu"},
            {"input_dim": 6, "output_dim": 6, "activation": "relu"},
            {"input_dim": 6, "output_dim": 4, "activation": "relu"},
            {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},]
        self.X = [[0,0],[0,1],[1,0],[1,1]]
        self.Y = [0,1,0,1]
        
    def init_layers(self, seed = 99):
        np.random.seed(seed)
        number_of_layers = len(self.nn_architecture)
        params_values = {}

        for idx, layer in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            params_values['W' + str(layer_idx)] = np.random.randn(
                layer_output_size, layer_input_size) * 0.1
            params_values['b' + str(layer_idx)] = np.random.randn(
                layer_output_size, 1) * 0.1

        return params_values

    def sigmiod(self):
        return 1/(1+np.exp(-self))

    def relu(self):
        return np.maximum(0,self)

    def sigmoid_backward(self, dA):
        sig = 1/(1+np.exp(-self))
        return dA * sig * (1 - sig)

    def relu_backward(self, dA):
        dZ = np.array(dA, copy = True)
        dZ[self <= 0] = 0;
        return dZ;

    def identity_function(self):
        return self
    



    def get_loss(self):

        Y_hat = self.forword_loss
        Y = self.Y
        m = Y_hat.shape[1]
        cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
        return np.squeeze(cost)

    def backward_propagation(X, params_values, nn_architecture):
        # creating a temporary memory to store the information needed for a backward step
        memory = {}
        # X vector is the activation for layer 0 
        A_curr = X
        
        # iteration over network layers
        for idx, layer in enumerate(nn_architecture):
            # we number network layers from 1
            layer_idx = idx + 1
            # transfer the activation from the previous iteration
            A_prev = A_curr
            
            # extraction of the activation function for the current layer
            activ_function_curr = layer["activation"]
            # extraction of W for the current layer
            W_curr = params_values["W" + str(layer_idx)]
            # extraction of b for the current layer
            b_curr = params_values["b" + str(layer_idx)]
            # calculation of activation for the current layer
            A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
            
            # saving calculated values in the memory
            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr
        
        # return of prediction vector and a dictionary containing intermediate values
        return A_curr, memory


a = Network()
x = np.array([1.0,1.0])
y = a.forword_propagation(x)
# test_loss = a.get_loss(1,y)
a.test()