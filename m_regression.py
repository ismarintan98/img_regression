import numpy as np


class nn_layers:

    def __init__(self,input_size,hidden_size):
        self.input_layer = np.zeros(input_size)
        self.hidden_layer = np.zeros(hidden_size)
        self.output_layer = np.zeros(hidden_size)
        self.gradient = np.zeros(hidden_size)

        self.weights = np.random.rand(input_size,hidden_size)

        self.bias = np.random.rand(hidden_size)

        

    def forward(self,input):
        self.input_layer = input
        self.hidden_layer = np.matmul(self.input_layer,self.weights) + self.bias

        #output layer sigmoid
        self.output_layer = 1/(1+np.exp(-self.hidden_layer))

        return self.output_layer

    def backward(self,learning_rate):
        
        #gradient sigmoid
        self.gradient = self.output_layer*(1-self.output_layer)

        return self.gradient






def main():

    #load banner.txt to banner
    banner = open("banner.txt","r")
    print(banner.read())

    layer1 = nn_layers(2,12)
    layer2 = nn_layers(12,1)

    input = [0,1]
    layer1.forward(input)
    layer2.forward(layer1.output_layer)

    layer2.backward(0.1)






if __name__ == '__main__':
    main()


