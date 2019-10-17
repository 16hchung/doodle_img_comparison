# resize image
from PIL import Image
from torchvision import transforms

class dataset():
    def __init__(self, path, save_path):
        # path only points to 1 image file
        self.image = Image.open(path)
        self.target_path = save_path

    def transform(self):
        resize = transforms.Resize(size=(128, 128))
        image = transforms.functional.rotate(resize(self.image), 90)
        #image.show()
        image_tensor = transforms.functional.to_tensor(image)
        print(image_tensor.size())
        return image_tensor

test = dataset('/Users/alexnam/Desktop/headshots/DSC_1036.JPG', '/Users/alexnam/Desktop/headshots/target_sample.jpg')
im_tensor = test.transform()
