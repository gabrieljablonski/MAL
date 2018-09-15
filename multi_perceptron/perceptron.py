from pickle import dump, load

from neuron import Neuron, ACTIVATED, NOT_ACTIVATED
from sources_target_label import SourcesTargetLabel

MAX_CYCLES = 1000


class TrainingFailedException(Exception):
    pass


class Perceptron:
    def __init__(self, labels=None, learning_rate: float = 1, activation_function: callable = None):
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.labels = labels

        if labels is not None:
            self.neurons = {label: Neuron(label, activation_function, learning_rate)
                            for label in labels}
        else:
            self.neurons = {}

        self.sources_target_label = []

        self.cycles = 0
        self.adjustments = 0

    def add_sources(self, target_label, sources):
        sources_target_label = SourcesTargetLabel(target_label, sources)
        self.sources_target_label.append(sources_target_label)

    def reset_weights(self, random_init=False):
        for neuron in self.neurons.values():
            neuron.initialize_weights(n_inputs=len(self.sources_target_label[0].sources), random_init=random_init)

    def output(self, inputs, which_label: str = None):
        if which_label is None:
            output = {neuron.specialties_label: neuron.output(inputs) for neuron in self.neurons.values()}
        else:
            neuron = self.neurons[which_label]
            output = {neuron.specialties_label: neuron.output(inputs)}

        return output

    # def adjust_weights(self, sources_target_label, which_neuron):
    #     self.adjustments += 1
    #     neuron = self.neurons[which_neuron]
    #     neuron.adjust_weights(sources_target_label)

    def train(self, random_starting_weights=False):
        self.reset_weights(random_starting_weights)
        self.cycles = 0

        while True:
            self.cycles += 1
            missed = False
            for neuron in self.neurons.values():
                for sources_target_label in self.sources_target_label:
                    s = sources_target_label
                    output = neuron.output(s.sources)
                    if (s.target_label in neuron.specialties_label and output == NOT_ACTIVATED) \
                            or (s.target_label not in neuron.specialties_label and output == ACTIVATED):
                        missed = True
                        print(f"Missed a {s.target_label} at a {neuron.specialties_label} neuron")
                        neuron.adjust_weights(s)
                        self.adjustments += 1

            if not missed:
                break
            print(f"Training cycle: #{self.cycles}")
            if self.cycles >= MAX_CYCLES:
                raise TrainingFailedException(f"Max cycle number ({MAX_CYCLES}) exceeded")

    def save_neurons(self, path):
        with open(path, 'wb') as file:
            dump(self.neurons, file)

    def load_neurons(self, path):
        with open(path, 'rb') as file:
            self.neurons = load(file)
