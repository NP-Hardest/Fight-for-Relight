import numpy as np
from PIL import Image
import cv2


def create_env():
    arr = np.zeros((32, 64, 3))
    for i in range(32):
        for j in range(32):
            245, 155, 66
            arr[i][j] = np.array([255, 0, 0])
            arr[i][j+32] = np.array([0, 0, 255])
            # arr[i][j+32] = np.array([0, 0, 0])

    # img = Image.fromarray(arr)
    # img.save("map.png")        
    Image.fromarray((arr).astype('uint8')).save("inputs/map2.png")


create_env()