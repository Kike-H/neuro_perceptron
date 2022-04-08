import numpy as np

class Perceptron:
    def __init__(self, size: int, alfa: np.number, num_epochs: int = 100):
        self.weights = np.random.randn(size)
        self.size = size
        self.num_epochs = num_epochs
        self.evo = [self.weights]
        self.alfa = alfa
        self.epoch = 0

    def activations(self, data:list ,tags: list) -> None:
        E = 0
        while True and self.epoch != self.num_epochs:
            self.epoch+=1
            for i,v in enumerate(data):
                y = (1*(self.weights.dot(v) > 0))
                if tags[i] != y:
                    self.__update(tags[i], y, v)
                E = E+tags[i]-y
            self.evo = np.concatenate((self.evo, [self.weights]), axis=0)
            print("e: ",self.epoch, "E: ", E)          
            if E == 0:
                break

        
    
    def __update(self, expected: int, o: int, inputs: list) -> None:
        for i in range(self.size):
            self.weights[i] = self.weights[i]+self.alfa*(expected-o)*(inputs[i])


