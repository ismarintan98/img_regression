
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(101)
tf.random.set_seed(101)

x = np.linspace(0,50,50)
y = np.linspace(0,50,50)

x += np.random.uniform(-4,4,50)
y += np.random.uniform(-4,4,50)

n = len(x)

# Plot of Training Data
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data")
plt.show()



X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(np.random.randn(), name = "W")
b = tf.Variable(np.random.randn(), name = "b")

