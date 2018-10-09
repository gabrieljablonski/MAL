from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from adaline import Adaline


DIGITS = 10
HOW_MANY = 1
BASE_PATH = f"digit_set_{HOW_MANY}ofeach/"


def load_image(path):
    return Image.open(path).convert(mode='L')


def image_to_bipolar_array(image):
    array = np.array([-1 if value == 255 else 1 for value in np.array(image).flatten()])

    return array


def main(train=False):
    digit_arrays = {
        str(digit): [load_image(f"{BASE_PATH}{digit}_{index}.png")
                     for index in range(HOW_MANY)]
        for digit in range(DIGITS)
    }

    flat_arrays = {
        digit: list(map(image_to_bipolar_array, images))
        for digit, images in digit_arrays.items()
    }

    learning_rate = 0.0005
    network = Adaline(labels=list(digit_arrays.keys()),
                      learning_rate=learning_rate)

    if train:
        for label, sources in flat_arrays.items():
            for source in sources:
                network.add_sources(source, label)

        max_cycles = 200
        random_starting_weights = False
        weight_adjustment_tolerance = None
        square_error_tolerance = None

        print(f"Max cycles: {max_cycles}\n--------------------------")
        network.train(random_starting_weights=random_starting_weights, max_cycles=max_cycles,
                      weight_adjustment_tolerance=weight_adjustment_tolerance,
                      square_error_tolerance=square_error_tolerance,
                      verbose=True)

        network.save_neurons('neurons.pkl')

    else:
        network.load_neurons('neurons.pkl')

    while True:
        # test_image = draw.get_character()
        test_image = load_image(input("image name:\n>>"))

        if test_image is None:
            break

        flat = image_to_bipolar_array(test_image)
        out = network.output(flat)

        for key, value in out.items():
            print(f"{key}: {value:.3f}")


if __name__ == '__main__':
    option = input("Train? (y/N)\n>>")

    main(train=True if option == 'y' else False)
