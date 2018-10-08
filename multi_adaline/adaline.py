from pickle import dump, load

from neuron import Neuron, ACTIVATED, NOT_ACTIVATED
from sources_target import SourcesTargetLabel

SQUARE_ERROR_DIVERGENCE = 10000


def default_activation_function(input_):
    output = input_
    return output


class Adaline:
    def __init__(self, labels=None, learning_rate: float = 0.1, activation_function: callable = None):
        self._learning_rate = learning_rate
        if activation_function is not None:
            self.activation_function = activation_function
        else:
            self.activation_function = default_activation_function

        if labels is not None:
            self.neurons = {label: Neuron(label, learning_rate, self.activation_function)
                            for label in labels}
        else:
            self.neurons = {}

        self.list_sources_target_label = []

        self.cycles = 0

        self.total_square_error_by_cycle = []

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

        for neuron in self.neurons.values():
            neuron.learning_rate = learning_rate

    def add_sources(self, sources, target_label):
        sources_target_label = SourcesTargetLabel(sources, target_label)
        self.list_sources_target_label.append(sources_target_label)

    def reset_weights(self, random_init=False):
        self.total_square_error_by_cycle = []
        n_inputs = len(self.list_sources_target_label[0].sources)

        for neuron in self.neurons.values():
            neuron.initialize_weights(n_inputs=n_inputs, random_init=random_init)

    def output(self, inputs, which_label: str = None):
        if which_label is None:
            output = {
                neuron.specialties_label: neuron.output(inputs)
                for neuron in self.neurons.values()
            }
        else:
            neuron = self.neurons[which_label]
            output = {neuron.specialties_label: neuron.output(inputs)}

        return output

    def train(self, random_starting_weights=False, max_cycles: int = None,
              weight_adjustment_tolerance: float = None, square_error_tolerance: float = None,
              verbose=True):
        self.reset_weights(random_starting_weights)
        self.cycles = 0

        while True:
            self.cycles += 1
            weight_adjustments = []
            total_square_error = 0
            for sources_target_label in self.list_sources_target_label:
                sources, target_label = sources_target_label.sources, sources_target_label.target_label

                for neuron in self.neurons.values():
                    output = neuron.output(sources)

                    if target_label in neuron.specialties_label:
                        target = ACTIVATED
                    else:
                        target = NOT_ACTIVATED
                    total_square_error += 0.5*(target - output)**2

                    bias_adjustment = neuron.adjust_bias(target_label, output)
                    weight_adjustments.append(bias_adjustment)

                    for index, source in enumerate(sources):
                        weight_adjustment = neuron.adjust_weight(index, source, target_label, output)
                        weight_adjustments.append(weight_adjustment)

            self.total_square_error_by_cycle.append(total_square_error/len(self.neurons))
            max_weight_adjustment = max(weight_adjustments)

            if verbose:
                print(f"Cycle: {self.cycles}")
                print(f"Square Error: {total_square_error}")
                print(f"Max adjustment: {max_weight_adjustment}")
                print("~~~")

            if total_square_error > SQUARE_ERROR_DIVERGENCE:
                print(f"Total square error diverging: {total_square_error:.5f}")
                break

            if max_cycles is not None and self.cycles >= max_cycles:
                print(f"Max cycles exceed")
                break

            if weight_adjustment_tolerance is not None and max_weight_adjustment <= weight_adjustment_tolerance:
                print(f"Maximum weight adjustment of {max_weight_adjustment:.5f}. "
                      f"(Threshold: f{weight_adjustment_tolerance:.5f}")
                break

            if square_error_tolerance is not None and total_square_error <= square_error_tolerance:
                print(f"Total square error of {total_square_error:.5f}. "
                      f"(Threshold: f{square_error_tolerance:.5f}")
                break

    def save_neurons(self, path):
        with open(path, 'wb') as file:
            dump(self.neurons, file)

    def load_neurons(self, path):
        with open(path, 'rb') as file:
            self.neurons = load(file)
