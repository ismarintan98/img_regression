import math

import matplotlib.pyplot as plt
import numpy as np

from nn_sin import nn_layers


def main():
    x1 = [-1, -0.5, 0, 0.5, 1]
    x2 = [-2, -1, 0, 1, 2]
    x3 = [-2.5, -1.5, 0, 1.5, 2.5]

    x_plot = [0, 1, 2, 3, 4]

    y = np.zeros((len(x1)))

    for i in range(len(y)):
        y[i] = x1[i]*x2[i]*x3[i] + np.sin(x1[i]+x2[i]+x3[i])

#    for i in range(5):
#     print(i,":",x1[i],x2[i],x3[i],y[i])

#    plt.plot(x_plot,y)
#    plt.show()

    layer_input = np.zeros((3, 1))
    # layer_input[0][0] = x1[0]
    # layer_input[1][0] = x2[0]
    # layer_input[2][0] = x3[0]

    layer_outputEnd = np.zeros((1, 1))
    # layer_outputEnd = y[0]

    layer1 = nn_layers(3, 12, 'sigmoid')
    layer2 = nn_layers(12, 1, 'purelin')

    learning_rate = 0.1


    x1_val = [0.5 , 0.4, 0.3, 0.2, 0.1]
    x2_val = [0.5 , 0.4, 0.3, 0.2, 0.1]
    x3_val = [0.5 , 0.4, 0.3, 0.2, 0.1]
    y_val = np.zeros((len(x1_val)))

    for i in range(len(y_val)):
        y_val[i] = x1_val[i]*x2_val[i]*x3_val[i] + np.sin(x1_val[i]+x2_val[i]+x3_val[i])
    
    print("Training...")
    for i in range(5000):
        for j in range(len(x1)):
            layer_input[0][0] = x1_val[j]
            layer_input[1][0] = x2_val[j]
            layer_input[2][0] = x3_val[j]
            layer_outputEnd[0][0] = y_val[j]

            layer1.forward(layer_input)
            layer2.forward(layer1.output_layer)

            layer2.backwardEndLayer(layer_outputEnd, learning_rate)
            layer1.backwardMidLayer(layer2, learning_rate)
            print("Epoch:",i)

    print("Training Complete")

    


    predict = np.zeros((len(x1)))

    for i in range(len(x1_val)):
        layer_input[0][0] = x1_val[i]
        layer_input[1][0] = x2_val[i]
        layer_input[2][0] = x3_val[i]

        layer1.forward(layer_input)
        layer2.forward(layer1.output_layer)

        predict[i] = layer2.output_layer[0][0]

        print("x1:",x1_val[i],"x2:",x2_val[i],"x3:",x3_val[i],"y:",y_val[i],"predict:",predict[i])

    
    RMSE = np.sqrt(np.mean((y_val-predict)**2)) #type: ignore

    print("RMSE:",RMSE)

    

    plt.plot(x_plot,y_val,'r',label='True')
    plt.plot(x_plot,predict,'b',label='Predict')    
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
