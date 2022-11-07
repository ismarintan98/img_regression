import numpy as np


class nn_layers:

    def __init__(self,input_size,hidden_size):
        self.input_layer = np.zeros(input_size)
        self.hidden_layer = np.zeros(hidden_size)
        self.output_layer = np.zeros(hidden_size)

        self.weights = np.random.rand(input_size,hidden_size)

        self.bias = np.random.rand(hidden_size)

        
        
        pass

    def forward(self,input):
        self.input_layer = input
        self.hidden_layer = np.matmul(self.input_layer,self.weights) + self.bias
        self.output_layer = self.hidden_layer

        return self.output_layer





def main():

    #load banner.txt to banner
    banner = open("banner.txt","r")
    print(banner.read())

    layer1 = nn_layers(2,2)
    layer1.weights[0]  = [1,2]
    layer1.weights[1]  = [3,4]
    
    layer1.bias[0] = 1
    layer1.bias[1] = 2
    

    input = [1,2]

    

    output = layer1.forward(input)

    print("------- Calc -------")
    print(np.shape(input))
    print(np.shape(layer1.weights))
    print(np.shape(layer1.bias))
    print(np.shape(layer1.hidden_layer))
    print(np.shape(layer1.output_layer))
    print("------- Result -------")
    print(output)
    # print("input:", input)
    # Calling the forward function in the nn_layers class.
    # print("output:", output)

    


if __name__ == '__main__':
    main()


