import numpy as np
import cv2

from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

class dataset():
    def __init__(self, path, save_path):
        # path only points to 1 image file
        self.image = Image.open(path).convert('L')
        self.target_path = save_path

    def transform(self, l=128, w=128):
        resize = transforms.Resize(size=(l, w))
#        image = transforms.functional.rotate(resize(self.image), 90)
        image = resize(self.image)
        image.show()
        raster =np.array(image, dtype='uint8')
        print(raster.shape)
        print(raster[0])
        print(raster[1])

        np.save(self.target_path + '.npy', raster)

        img_array = np.load(self.target_path + '.npy')
        plt.imshow(img_array)
    #   image_tensor = transforms.functional.to_tensor(image)
    #    print(image_tensor.size())
    #    return image_tensor

test = dataset('/Users/alexnam/Desktop/bear_image.png', '/Users/alexnam/Desktop/target_sample')
test.transform(l=28, w=28)



def resize(load_path, save_path, l=14, w=14):
    image = np.load(load_path)
    res = cv2.resize(image, dsize=(l, w), interpolation=cv2.INTER_CUBIC)
    np.save(save_path, res)


PATH = 'full_numpy_bitmap_'
SAVEPATH = 'resized_numpy_'
#resize(PATH + 'airplane.jpg', SAVEPATH + 'airplane_image.npy')
#resize(PATH + 'apple.jpg', SAVEPATH + 'apple_image.npy')
#resize(PATH + 'bear.jpg', SAVEPATH + 'bear_image.npy')

