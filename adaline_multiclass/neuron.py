import random

WEIGHT_INIT_LIMITS = (-0.001, 0.001)

ACTIVATED = 1
NOT_ACTIVATED = -1
DEFAULT_ACTIVATION = ACTIVATED


class Neuron:
    def __init__(self, specialties_label: str, learning_rate: float = 0.1, activation_function: callable = None):
        self.specialties_label = specialties_label
        self.learning_rate = learning_rate
        self.activation_function = activation_function

        self.weights, self.starting_weights = None, None
        self.bias, self.starting_bias = None, None

    def initialize_weights(self, n_inputs, random_init=False):
        self.weights = [random.uniform(*WEIGHT_INIT_LIMITS) if random_init else 0
                        for _ in range(n_inputs)]
        self.bias = random.uniform(*WEIGHT_INIT_LIMITS) if random_init else 0

        self.starting_weights = self.weights
        self.starting_bias = self.bias

    def output(self, inputs):
        net_value = self.bias
        for input_, weight in zip(inputs, self.weights):
            net_value += weight * input_

        output = self.activation_function(net_value)
        return output

    def adjust_bias(self, target_label, output):
        if target_label is self.specialties_label:
            target = ACTIVATED
        else:
            target = NOT_ACTIVATED

        old_bias = self.bias
        new_bias = old_bias + self.learning_rate*DEFAULT_ACTIVATION*(target - output)
        adjustment = abs(old_bias-new_bias)

        self.bias = new_bias

        return adjustment

    def adjust_weight(self, which_weight: int, source, target_label, output):
        if target_label is self.specialties_label:
            target = ACTIVATED
        else:
            target = NOT_ACTIVATED

        old_weight = self.weights[which_weight]
        new_weight = old_weight + self.learning_rate*source*(target-output)
        adjustment = abs(old_weight-new_weight)

        self.weights[which_weight] = new_weight

        return adjustment
