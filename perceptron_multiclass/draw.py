import cv2
import numpy as np

ESC_KEY = 27
ENTER_KEY = 13

COLOR = (255, 255, 255)
THICKNESS = 3

DRAW_SHAPE = (150, 150, 3)
TARGET_SHAPE = (50, 50)

WINDOW_NAME = 'Drawing'


class Drawing:
    def __init__(self):
        self.img = None
        self.drawing = False

        self.last_pos = (0, 0)
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.interactive_drawing)

    def interactive_drawing(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True

        elif event == cv2.EVENT_MOUSEMOVE:
            new_pos = (x, y)
            if self.drawing:
                cv2.line(img=self.img, pt1=self.last_pos, pt2=new_pos,
                         color=COLOR, thickness=THICKNESS)
            self.last_pos = new_pos

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def reset_img(self):
        self.img = np.zeros(DRAW_SHAPE, np.uint8)

    def get_character(self):
        self.reset_img()
        while True:
            cv2.imshow(WINDOW_NAME, self.img)

            key = cv2.waitKey(1) & 0xFF
            if key == ESC_KEY:
                # self.reset_img()
                # continue
                return None
            if key == ENTER_KEY:
                self.img = cv2.resize(self.img, TARGET_SHAPE)
                return self.img

    def save_character(self, path):
        cv2.imwrite(path, self.img)

    @staticmethod
    def destroy_window():
        cv2.destroyWindow(WINDOW_NAME)


if __name__ == "__main__":
    for digit in range(10):
        which_digit = str(digit)

        base_path = f"digit_set_10ofeach/{which_digit}_"
        draw = Drawing()
        for index in range(10):
            draw.get_character()
            path = f"{base_path}{index}.png"
            draw.save_character(path)
            print(which_digit, index)
