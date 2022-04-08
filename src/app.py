from modules.perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    perceptron = Perceptron(3, 0.5)
    inputs = [[0,0,1,0],[0,1,1,0],[1,0,1,0],[1,1,1,1]]
    data = []
    tags = []


    for v in inputs:
        data.append(v[0:3])
        tags.append(v[3])

    print(data)
    print(tags)
    perceptron.activations(data, tags)
    plt.plot(perceptron.evo[:,0], 'k')
    plt.plot(perceptron.evo[:,1], 'r')
    plt.plot(perceptron.evo[:,2], 'b')

    # plt.savefig('./test.png')
    plt.show()




