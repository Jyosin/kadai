import numpy as np
class Network():
    def __init__(self):
        self.network = dict()
        self.network['W1'] = np.ones([2,3], dtype = float)
        self.network['b1'] = np.array([1.0,1.0,1.0])
        self.network['W2'] = np.ones([3,2], dtype = float)
        self.network['b2'] = np.array([1.0,1.0])
        self.network['W3'] = np.ones([2,1], dtype = float)
        self.network['b3'] = np.array([1.0])

    def sigmiod(self):
        return 1/(1+np.exp(-self))

    def relu(self):
        return np.maximum(0,self)

    def sigmoid_backward(dA, self):
        sig = 1/(1+np.exp(-self))
        return dA * sig * (1 - sig)

    def relu_backward(dA, self):
        dZ = np.array(dA, copy = True)
        dZ[self <= 0] = 0;
        return dZ;

    def identity_function(self):
        return self
    

    def forword_propagation(self,x):
        W1,W2,W3 = self.network['W1'],self.network['W2'],self.network['W3']
        b1,b2,b3 = self.network['b1'],self.network['b2'],self.network['b3']
        a1 = np.dot(x,W1)+b1
        z1 = Network.sigmiod(a1)
        a2 = np.dot(z1,W2)+b2
        z2 = Network.sigmiod(a2)
        a3 = np.dot(z2,W3) + b3
        z3 = Network.relu(a3)
        y = Network.identity_function(z3)
        return y

    self.loss = self.forword_propagation(self,x)

    def get_loss(forword_propagation(self,x),y):
        m = self.shape[1]
        cost = -1 / m * (np.dot(y, np.log(self).T) + np.dot(1 - y, np.log(1 - self).T))
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
y = a.forword(x)
y = 
print(y)