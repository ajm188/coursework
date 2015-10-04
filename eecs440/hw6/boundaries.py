from __future__ import division

import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random


def sigmoid(w, x):
    return 1 / (1 + math.exp(-np.vdot(w, x)))


def random_weights(lower, upper, N):
    samples = numpy.random.random(size=N)
    return (samples * (upper - lower)) + lower


def input_bounds():
    return xrange(-5, 5)


class FullyConnectedANN(object):

    class Node(object):

        def __init__(self, W,):
            self.W = W

        def activation(self, X):
            return sigmoid(self.W, X)

    def __init__(self,
                 input_layer_size,
                 hidden_layer_size,
                 weight_lower_bound,
                 weight_upper_bound):
        self.hidden_layer = []
        for i in xrange(hidden_layer_size):
            self.hidden_layer.append(
                FullyConnectedANN.Node(
                    random_weights(
                        weight_lower_bound,
                        weight_upper_bound,
                        input_layer_size,
                    ),
                ),
            )

        self.output_node = FullyConnectedANN.Node(
            random_weights(
                weight_lower_bound,
                weight_upper_bound,
                hidden_layer_size,
            ),
        )

    def propagate(self, X):
        intermediates = [hidden.activation(X) for hidden in self.hidden_layer]
        return self.output_node.activation(intermediates)


if __name__ == '__main__':
    weight_bounds = [10, 3, 0.1]
    for b in weight_bounds:
        ann = FullyConnectedANN(2, 2, -b, b)
        points = []
        for x_1 in input_bounds():
            for x_2 in input_bounds():
                output = ann.propagate(np.array([x_1, x_2]))
                color = 'g' if output >= 0.5 else 'r'
                points.append((x_1, x_2, color))
        x_1, x_2, colors = zip(*points)
        plt.scatter(x_1, x_2, c=colors)
        plt.title(
            "Decision Boundary for Random Weights in [%.1f, %.1f)" % (-b, b),
        )
        plt.show()
