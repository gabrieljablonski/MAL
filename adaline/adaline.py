from neuron import Neuron
from sources_target import SourcesTarget


class Adaline:
    def __init__(self, learning_rate: float = 0.1, activation_function: callable = None):
        self._learning_rate = learning_rate
        self.activation_function = activation_function

        self.neuron = Neuron(learning_rate, activation_function)

        self.list_sources_target = []

        self.cycles = 0

        self.total_square_error_by_cycle = []

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate
        self.neuron.learning_rate = learning_rate

    def add_sources(self, sources, target):
        sources_target = SourcesTarget(sources, target)
        self.list_sources_target.append(sources_target)

    def reset_weights(self, random_init=False):
        self.total_square_error_by_cycle = []
        n_inputs = len(self.list_sources_target[0].sources)
        self.neuron.initialize_weights(n_inputs=n_inputs, random_init=random_init)

    def output(self, inputs):
        output = self.neuron.output(inputs)
        return output

    def train(self, random_starting_weights=False, max_cycles: int = None,
              weight_adjustment_tolerance: float = None, square_error_tolerance: float = None):
        self.reset_weights(random_starting_weights)
        self.cycles = 0

        while True:
            self.cycles += 1
            weight_adjustments = []
            total_square_error = 0
            for sources_target in self.list_sources_target:
                sources, target = sources_target.sources, sources_target.target
                output = self.neuron.output(sources)
                total_square_error += 0.5*(target-output)**2

                bias_adjustment = self.neuron.adjust_bias(target, output)
                weight_adjustments.append(bias_adjustment)

                for index, source in enumerate(sources):
                    weight_adjustment = self.neuron.adjust_weight(index, source, target, output)
                    weight_adjustments.append(weight_adjustment)

            self.total_square_error_by_cycle.append(total_square_error)

            if total_square_error > 100:
                break

            if max_cycles is not None and self.cycles >= max_cycles:
                break

            if weight_adjustment_tolerance is not None and max(weight_adjustments) <= weight_adjustment_tolerance:
                break

            if square_error_tolerance is not None and total_square_error <= square_error_tolerance:
                break
