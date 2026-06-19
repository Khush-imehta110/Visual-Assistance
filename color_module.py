import cv2
import numpy as np

# Standard HSV ranges
color_ranges = {
    "Pink": [(140, 20, 50), (170, 255, 255)],
    "Magenta": [(125, 20, 50), (150, 255, 255)],
    "Lavender": [(110, 20, 80), (140, 255, 255)],

    "Green": [(35, 50, 50), (85, 255, 255)],
    "White": [(0, 0, 200), (255, 40, 255)],
    "Black": [(0, 0, 0), (180, 255, 40)],
}


def detect_color(image):
    """
    Detects dominant colors in the input image based on HSV ranges.
    Returns a string describing colors that appear more than threshold %.
    """

    # Resize image for consistency
    img = cv2.resize(image, (640, 640))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color_percentage = {}

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        ratio = (cv2.countNonZero(mask) / (img.size / 3)) * 100
        color_percentage[color] = ratio

    # Prepare readable output
    detected = [f"{color}" for color, perc in color_percentage.items() if perc > 0.2]

    if not detected:
        return "No significant color detected."

    return "Detected colors: " + ", ".join(detected)

