import sys
import math
import pandas as pd


def main():
    # get arguments from command line and set them
    """ (1) the path to a training file, (2) the path to a test file, (3) the number
    of hidden layers, (4) the number of hidden nodes per layer, (5) a learning rate, and (6) the
    number of iterations to run the algorithm"""

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    hidden_layers = sys.argv[3]
    hidden_nodes = sys.argv[4]
    learning_rate = sys.argv[5]
    iterations = sys.argv[6]

    train = pd.read_csv(train_file, delimiter='\t', header=0)
    test = pd.read_csv(test_file, delimiter='\t', header=0)
    
    input = (train.iloc[0].values)[:-1]
    
    net = Network(int(hidden_layers), int(hidden_nodes), float(learning_rate), len(input))
    net.train(train, test, int(iterations))
    


def sig(input):
    return 1 / (1 + math.exp(-input))


class Node:
    def __init__(self, input_size, learningRate, value):
        self.weights = ([0] * input_size)
        self.value = value
        self.delta = 0
        self.learning_rate = learningRate
        self.target = 0

    def __repr__(self):
        return f"Node: weights = {self.weights} value = {self.value} delta = {self.delta}"

    # setter functions
    def set_value(self, value):
        self.value = value

    def set_target(self, target):
        self.target = target

    # as the network advances the forward pass sums all weight*input per i
    def forward(self, prev_layer):
        sum = 0
        for i, node in enumerate(prev_layer.nodes):
            sum += node.value * self.weights[i]
        self.value = sig(sum)

    def backward(self, curr_node_index, next_layer=None):
        # calculate node's delta based on last layer's deltas
        if next_layer is None:
            # if current layer is output layer get real error
            self.delta = self.value * (1 - self.value) * (self.target - self.value)
        else:
            self.delta = self.value * (1 - self.value) * sum([node.weights[curr_node_index] * node.delta for node in next_layer.nodes])

    def update_weights(self, prev_layer):
        # update weights based on new delta
        for i, (weight, input) in enumerate(zip(self.weights, prev_layer.nodes)):
            self.weights[i] = weight + self.learning_rate * self.delta * input.value


class Layer:
    def __init__(self, num_nodes, input_size, learningRate, add_bias):
        self.size = num_nodes
        if add_bias:
            self.nodes = [Node(input_size + 1, learningRate, 1)] + [Node(input_size + 1, learningRate, 0) for _ in range(num_nodes)]
        else:
            self.nodes = [Node(input_size + 1, learningRate, 0) for _ in range(num_nodes)]
        self.learning_rate = learningRate
    def __str__(self):
        return f"{self.nodes}"

    # set all values in a layer
    def set_values(self, values):
        for i, _ in enumerate(self.nodes):
            if i == 0:
                continue
            self.nodes[i].set_value(values[i-1])

    # set output node's target
    def set_target(self, target):
        self.nodes[0].set_target(target)

    # forward propagate layer
    def forward_prop(self, prev_layer, next_layer):
        for i, node in enumerate(self.nodes):
            if i == 0 and next_layer is not None:
                continue
            node.forward(prev_layer)

    def backward_prop(self, next_layer):
        for i, node in enumerate(self.nodes):
            # print(self)
            node.backward(i, next_layer)

    def update_w(self, prev_layer):
        for node in self.nodes:
            node.update_weights(prev_layer)


class Network:
    def __init__(self, num_hidden_layers, num_hidden_nodes, learn_rate, input_size):
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_nodes = num_hidden_nodes
        self.learn_rate = learn_rate
        self.layers = []
        self.input_size = input_size
        self.build_net()

    def __repr__(self):
        message = f""
        for i, layer in enumerate(self.layers):
            if layer is not None:
                message += f"layer {i}: {layer.nodes}\n"
        return message

    def add_input_layer(self):
        self.layers.append(Layer(self.input_size, 0, 0, True))

    def add_hidden_layer(self):
        self.layers.append(Layer(self.num_hidden_nodes, self.layers[-1].size, self.learn_rate, True))

    def add_output_layer(self):
        self.layers.append(Layer(1, self.layers[-1].size, self.learn_rate, False))
        # add a none at the end to denote output layer
        self.layers.append(None)

    def build_net(self):
        self.add_input_layer()
        for i in range(self.num_hidden_layers):
            self.add_hidden_layer()
        self.add_output_layer()

    def inference(self, input):
        # set input and classification
        self.layers[0].set_values(input)
        # forward propagate
        for i in range(1, len(self.layers) - 1):
            self.layers[i].forward_prop(self.layers[i - 1], self.layers[i + 1])
        # return output of last layer
        return self.layers[-2].nodes[0].value

    def train_instance(self, input, target, iteration, train_df, test_df):
        # set input and classification
        self.layers[0].set_values(input)
        self.layers[-2].set_target(target)
        # forward propagate
        for i in range(1, len(self.layers) - 1):
            self.layers[i].forward_prop(self.layers[i - 1], self.layers[i + 1])

        print(f"At iteration {iteration}")
        print(f"Forward pass output: {round(self.layers[-2].nodes[0].value, 4):.4f}")

        # back propagate
        for i in range(len(self.layers) - 2, 0, -1):
            self.layers[i].backward_prop(self.layers[i + 1])
        # update weights
        for i in range(len(self.layers) - 2, 0, -1):
            self.layers[i].update_w(self.layers[i - 1])

        # print(self)
        print(f"Average squared error on training set ({len(train_df)} instances): {round(avg_squared_error(train_df, self), 4):.4f}")
        print(f"Average squared error on test set ({len(test_df)} instances): {round(avg_squared_error(test_df, self), 4):.4f}\n")

    def train(self, input_df, test_df, iterations):
        iteration = 1
        while(True):
            for j in range(len(input_df)):
                if iteration > iterations:
                    return
                input = (input_df.iloc[j].values[:-1]).tolist()
                target = (input_df.iloc[j].values[-1])
                self.train_instance(input, target, iteration, input_df, test_df)
                iteration += 1
            # print(f"At iteration {i}")
            # print(f"Forward pass output: {self.layers[-2].nodes[0].value}")
            # print(f"Average squared error on training set ({len(input_df)} instances): {avg_squared_error(input_df, self)}")
            # print(f"Average squared error on test set ({len(test_df)} instances): {avg_squared_error(test_df, self)}\n")
            

def avg_squared_error(df, net):
    sum = 0
    for i in range(len(df)):
        input = (df.iloc[i].values[:-1]).tolist()
        target = (df.iloc[i].values[-1])
        sum += (net.inference(input) - target) ** 2
    return sum / len(df)


if __name__ == "__main__":
    main()