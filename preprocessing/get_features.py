import glob
import numpy as np
from PIL import Image
import PIL.ImageOps
import cv2
from torchvision import transforms
from matplotlib import pyplot as plt

DOODLE_PATH = 'downloads/full_numpy_bitmap_'
DOODLE_NUMPY_PATH = 'downloads/resized_numpy_'
def process_doodles(categories, d_w, d_h):
    doodle_files = []
    for c in categories:
        image = np.load(DOODLE_PATH + c + '.npy')
        full = []
        for i in range(len(image)):
            im = np.reshape(np.invert(image[i]), (28, 28))
            res = cv2.resize(im, dsize=(d_w, d_h), interpolation=cv2.INTER_CUBIC)
            v = np.reshape(res, (d_w * d_h,))
            full.append(v)
        doodle_file_name = DOODLE_NUMPY_PATH + c + '.npy'
        np.save(doodle_file_name, full)
        doodle_files.append(doodle_file_name)
    return doodle_files

IMG_PATH = 'google-images-download/downloads/'
IMG_NUMPY_PATH = 'downloads/resized_numpy_'
def process_images(categories, img_w, img_h):
    img_files = []
    for c in categories:
        full = []
        for img_file in glob.glob(IMG_PATH + c + '/' + '*.jpg'):
            img = Image.open(img_file).convert('L')
            img_resized = img.resize((img_w, img_h), Image.ANTIALIAS)
            img_arr = np.array(img_resized, dtype=float)
            v = np.reshape(img_arr, (img_w * img_h,))
            full.append(v)
        img_file_name = IMG_NUMPY_PATH + c + '.npy'
        np.save(img_file_name, full)
        img_files.append(img_file_name)
    return img_files

def create_features(categories, zero, num_labels):
    aug = []
    img_dic = {}
    doodle_dic = {}
    true_labels = []
    false_labels = []
    X = []
    y = []
    for c in categories:
        img = np.load(IMG_NUMPY_PATH + c + '.npy')
        img_dic[c] = img
        doodle = np.load(DOODLE_NUMPY_PATH + c + '.npy')
        doodle_dic[c] = doodle
    
    num_labels_per_example = int(num_labels/float(len(categories)))
    # false labels: (vectorized img + doodle, -1/0)
    for img_key1 in img_dic.keys():
        img = img_dic[img_key1]
        for img_key2 in img_dic.keys():
            if img_key1 != img_key2:
                doodle = doodle_dic[img_key2]
                r_doodle = np.random.randint(len(doodle), size=num_labels_per_example)
                r_img = np.random.randint(len(img), size=num_labels_per_example)
                for r in range(len(r_img)):
                    X.append(np.concatenate((img[r_img[r]], doodle[r_doodle[r]]), axis=None))
                    if zero:  
                        y.append(0)
                    else: 
                        y.append(-1)

    # true labels: (vectorized img + doodle, 1) 
    for img_key in img_dic.keys():
        img = img_dic[img_key]
        doodle = doodle_dic[img_key]
        r_img = np.random.randint(len(img), size=num_labels)
        r_doodle = np.random.randint(len(doodle), size=num_labels)
        for j in range(len(r_img)):
            X.append(np.concatenate((img[r_img[j]], doodle[r_doodle[j]]), axis=None))
            y.append(1)   
    np.save('downloads/' + 'X', X)
    np.save('downloads/' + 'y', y)
    return X, y


# image size (img_w, img_h), doodle size (d_w, d_h), # of labels, -1/0 for y
# categories = {'apple', 'bear', 'tree'} list of strings
def generate_features(categories, img_w, img_h, d_w, d_h, num_labels, zero):
    doodle_files = process_doodles(categories, d_w, d_h)
    img_files = process_images(categories, img_w, img_h)
    X, y = create_features(categories, zero, num_labels)

generate_features({'apple', 'bear', 'airplane'}, 14, 14, 14, 14, 10, True)
