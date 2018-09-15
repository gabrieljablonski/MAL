import random
from numpy import ndarray as array

ACTIVATED = 1
NOT_ACTIVATED = -1
DEFAULT_ACTIVATION = ACTIVATED

STARTING_WEIGHT_LIMITS = (-0.5, 0.5)
MAX_CYCLES = 100


class TrainingFailedException(Exception):
    pass


class SourcesTarget:
    def __init__(self, sources, target):
        self.sources = sources
        self.target = target

    def __len__(self):
        return len(self.sources)


class Perceptron:
    def __init__(self, learning_rate: float = 1, activation_function: callable = None):
        self.learning_rate = learning_rate
        self.activation_function = activation_function

        self.sources_targets = []
        self._weights = []
        self._bias = 0
        self.starting_weights = []
        self.starting_bias = 0
        self.cycles = 0

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        raise PermissionError("Weights cannot be manually set")

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        raise PermissionError("Weights cannot be manually set")

    def add_sources(self, sources: array, target: int):
        try:
            target = int(target)
        except Exception:
            raise TypeError("Target must be integer")

        if target not in (ACTIVATED, NOT_ACTIVATED):
            raise ValueError("Invalid value for target")
        self.sources_targets.append(SourcesTarget(sources, target))

    def reset_weights(self, random_start=False):
        self._weights = [random.uniform(*STARTING_WEIGHT_LIMITS)
                        if random_start else 0 for _ in range(len(self.sources_targets[0]))]
        self._bias = random.uniform(*STARTING_WEIGHT_LIMITS) if random_start else 0

        self.starting_weights = self.weights
        self.starting_bias = self.bias

    def output(self, inputs: array):
        net_value = self._bias
        for input_, weight in zip(inputs, self._weights):
            net_value += weight * input_

        output = self.activation_function(net_value)
        try:
            output = int(output)
        except Exception:
            raise TypeError("Output from activation function must be integer")

        if output not in (ACTIVATED, NOT_ACTIVATED):
            raise ValueError("Output from activation function must be 1 or -1")

        return output

    def update_weights(self, sources_target: SourcesTarget):
        new_weights = []
        s = sources_target
        for source, target, old_weight in zip(s.sources, s.target, self._weights):
            new_weight = old_weight + self.learning_rate*target*source
            new_weights.append(new_weight)

        self._bias += DEFAULT_ACTIVATION * self.learning_rate*s.target
        self._weights = new_weights

    def train(self, random_starting_weights=False):
        self.reset_weights(random_starting_weights)
        self.cycles = 0

        while True:
            self.cycles += 1
            missed = False
            for sources_target in self.sources_targets:
                output = self.output(sources_target.sources)
                if output != sources_target.target:
                    missed = True
                    self.update_weights(sources_target)

            if not missed:
                break
            if self.cycles >= MAX_CYCLES:
                raise TrainingFailedException(f'Max cycle number ({MAX_CYCLES}) exceeded')
