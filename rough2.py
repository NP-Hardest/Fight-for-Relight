import numpy as np
from PIL import Image


def load_image(path):
    img = Image.open(path)
    return np.array(img) / 255.0

def save_image(image, path):
    Image.fromarray((image * 255).astype('uint8')).save(path)




def extract_bg(inp, al):
    input = load_image(inp)
    print(input.shape)
    alpha = load_image(al)

    height, width = input.shape

    arr = np.zeros((height, width, 3))

    for y in height:
        for x in width:
            arr[y, x] = (input[y, x][:3]) * (1 - alpha[y, x])

    arr = np.clip(arr, 0, 1)
    save_image(arr, "bg.png")


    
extract_bg(inp = "inputs/in.png", al = "inputs/alpha.png")

