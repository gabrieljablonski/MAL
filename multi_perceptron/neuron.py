import random
from sources_target_label import SourcesTargetLabel

WEIGHT_INIT_LIMITS = (-0.5, 0.5)

ACTIVATED = 1
NOT_ACTIVATED = -1
DEFAULT_ACTIVATION = ACTIVATED


class Neuron:
    def __init__(self, specialties_label: str, activation_function, learning_rate: float = 1):
        self.specialties_label = specialties_label
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

        self.starting_weights = []
        self.starting_bias = []

    def initialize_weights(self, n_inputs, random_init=False):
        if n_inputs > 0:
            self.weights = [random.uniform(*WEIGHT_INIT_LIMITS) if random_init else 0 for _ in range(n_inputs)]
            self.bias = random.uniform(*WEIGHT_INIT_LIMITS) if random_init else 0

            self.starting_weights = self.weights
            self.starting_bias = self.bias
        else:
            raise ValueError("Invalid number of inputs")

    def output(self, inputs):
        net_value = self.bias
        for input_, weight in zip(inputs, self.weights):
            net_value += weight * input_

        output = self.activation_function(net_value)
        return output

    def adjust_weights(self, sources_target_label: SourcesTargetLabel):
        s = sources_target_label

        if s.target_label is self.specialties_label:
            target = ACTIVATED
        else:
            target = NOT_ACTIVATED

        new_weights = []
        for source, old_weight in zip(s.sources, self.weights):
            new_weight = old_weight + self.learning_rate*target*source
            new_weights.append(new_weight)
        self.weights = new_weights
        self.bias += self.learning_rate*target*DEFAULT_ACTIVATION
