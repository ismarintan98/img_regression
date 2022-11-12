import math

import numpy as np
import matplotlib.pyplot as plt


class nn_layers:

    def __init__(self,input_size,hidden_size,activation):
        self.input_layer = np.zeros((input_size,1))
        self.hidden_layer = np.zeros((hidden_size,1))
        self.output_layer = np.zeros((hidden_size,1))
        self.gradient = np.zeros(hidden_size)

        self.weights = np.random.rand(hidden_size,input_size)

        self.bias = np.random.rand(hidden_size,1)
        self.activation = activation
        self.error  = np.zeros((hidden_size,1))
        self.sensitivity = np.zeros((hidden_size,1))
        self.buffMatrix = np.zeros((hidden_size,hidden_size))



        

    def forward(self,input):
        self.input_layer = input
        self.hidden_layer = np.dot(self.weights,self.input_layer) + self.bias

        #output layer sigmoid
        if self.activation == 'sigmoid':        
            self.output_layer = 1/(1+np.exp(-self.hidden_layer))

        #purelin
        if self.activation == 'purelin':
            self.output_layer = self.hidden_layer

        return self.output_layer

    def backwardEndLayer(self,true_output,learning_rate):
        
        self.error = true_output - self.output_layer

        if self.activation == 'sigmoid':
            self.sensitivity = -2 *  self.output_layer * (1-self.output_layer) * self.error

        elif self.activation == 'purelin':
            self.sensitivity = -2  * 1 * self.error

        self.weights = self.weights - learning_rate * np.dot(self.sensitivity,np.transpose(self.input_layer))
        self.bias = self.bias - learning_rate * self.sensitivity

    

    def backwardMidLayer(self,next_layer,learning_rate):
        #gradient sigmoid

        if self.activation == 'sigmoid':
            self.gradient = self.output_layer * (1-self.output_layer)
            self.sensitivity = np.dot(next_layer.weights.T,next_layer.sensitivity) * self.gradient
        elif self.activation == 'purelin':
            self.gradient = 1
            self.sensitivity = np.dot(next_layer.weights.T,next_layer.sensitivity) * self.gradient

        self.weights = self.weights - learning_rate * np.dot(self.sensitivity,np.transpose(self.input_layer))
        self.bias = self.bias - learning_rate * self.sensitivity


    

        

    
        

        






def main():

    #load banner.txt to banner
    banner = open("banner.txt","r")
    print(banner.read())

    input = [1]
    true_output =  1 + math.sin((math.pi*input[0])/4)

    input_train = np.linspace(-3,3,100)
    true_output_train = 1 + np.sin((math.pi*input_train)/4)

    # for i in range(len(input_train)):
    #     true_output_train[i] = 1 + math.sin((math.pi*input_train[i])/4)

   

    

    layer1 = nn_layers(1,2,'sigmoid')
    layer2 = nn_layers(2,1,'purelin')

    learning_rate = 0.1

    

    layer1.weights[0][0] = -0.27
    layer1.weights[1][0] = -0.41
    layer1.bias[0][0] = -0.48
    layer1.bias[1][0] = -0.13

    layer2.weights[0][0] = 0.09
    layer2.weights[0][1] = -0.17
    layer2.bias[0][0] = 0.48



    val_input = np.linspace(-2,2,100)
    val_output = np.zeros((len(val_input)))
    val_pred = np.zeros((len(val_input)))
    for i in range(len(val_input)):
        val_output[i] = 1 + math.sin((math.pi*val_input[i])/4)


    print("Training...")
    for i in range(50000):
        for j in range(len(input_train)):
            
            # print("Epoch: ",i," Input: ",input_train[j]," True Output: ",true_output_train[j])
            layer1.forward(np.array([[input_train[j]]]))
            layer2.forward(layer1.output_layer)
            layer2.backwardEndLayer(np.array([[true_output_train[j]]]),learning_rate)
            layer1.backwardMidLayer(layer2,learning_rate)


    for i in range(len(val_input)):
        layer1.forward(np.array([[val_input[i]]]))
        layer2.forward(layer1.output_layer)
        val_pred[i] = layer2.output_layer[0][0]


    rmse = np.sqrt(np.mean((val_output-val_pred)**2))  # type: ignore
    print("RMSE: ",rmse)

    plt.plot(val_input,val_output,label="true",color="red")
    plt.plot(val_input,val_pred,label="predicted",color="blue")
    plt.legend()
    plt.show()

    





    


    

    

    

        






if __name__ == '__main__':
    main()


