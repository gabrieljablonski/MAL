from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from perceptron import Perceptron, ACTIVATED, NOT_ACTIVATED
from draw import Drawing

LEARNING_RATE = 0.1
THRESHOLD = 0

DIGITS = 10
HOW_MANY = 10

BASE_PATH = 'digit_set_10ofeach/'
NEURONS_PICKLE_PATH = 'neurons000.pkl'


def load_image(path):
    return Image.open(path).convert(mode='L')


def image_to_bipolar_array(image):
    array = np.array([-1 if value == 255 else 1 for value in np.array(image).flatten()])

    return array


def activation_function(input_):
    if input_ >= THRESHOLD:
        output = ACTIVATED
    else:
        output = NOT_ACTIVATED

    return output


def main(train=False):
    if train:
        digit_arrays = {str(digit): [load_image(f"{BASE_PATH}{digit}_{index}.png")
                                     for index in range(HOW_MANY)]
                        for digit in range(DIGITS)}

        flat_arrays = {digit: [image_to_bipolar_array(image)
                               for image in images]
                       for digit, images in digit_arrays.items()}

        network = Perceptron(labels=list(digit_arrays.keys()),
                             learning_rate=LEARNING_RATE,
                             activation_function=activation_function)

        for label, sources in flat_arrays.items():
            for source in sources:
                network.add_sources(label, source)

        network.train(random_starting_weights=True)
        network.save_neurons(NEURONS_PICKLE_PATH)

        print(f"Cycles: {network.cycles}")
        print(f"Adjustments: {network.adjustments}\n")

    else:
        network = Perceptron()
        network.load_neurons(NEURONS_PICKLE_PATH)

    draw = Drawing()

    while True:
        test_image = draw.get_character()
        # test_image = load_image(f"{BASE_PATH}0_0.png")

        if test_image is None:
            break

        flat = image_to_bipolar_array(test_image)
        out = network.output(flat)

        predict = [digit for digit, output in out.items() if output == ACTIVATED]

        if len(predict) == 0:
            out_string = "The network doesn't know what digit that is."
        elif len(predict) == 1:
            out_string = f"The network thinks that's a {predict[0]}."
        else:
            out_string = f"The network thinks that might be one of these digits: {predict}."
        print(out_string)


if __name__ == '__main__':
    main(train=False)
