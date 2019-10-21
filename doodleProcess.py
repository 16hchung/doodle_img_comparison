import numpy as np
import cv2

def resize(load_path, save_path, l=14, w=14):
    image = np.load(load_path)
    res = cv2.resize(image, dsize=(l, w), interpolation=cv2.INTER_CUBIC)
    np.save(save_path, res)


PATH = 'full_numpy_bitmap_'
SAVEPATH = 'resized_numpy_'
resize(PATH + 'airplane.npy', SAVEPATH + 'airplane.npy')
resize(PATH + 'apple.npy', SAVEPATH + 'apple.npy')
resize(PATH + 'bear.npy', SAVEPATH + 'bear.npy')

