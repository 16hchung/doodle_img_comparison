import glob
from pathlib import Path
import numpy as np
from PIL import Image
import PIL.ImageOps
import cv2
import torch
from torchvision import transforms
import torchvision.models as models
from matplotlib import pyplot as plt
from tqdm import tqdm

from util.constants import *
from .images_data import test_train_split

def process_doodles(categories, d_w, d_h, store_vgg):
    '''
    invert and resize doodles into specified dims and save to PROCESSED_DOODLE_PATH
    '''
    doodle_files = []
    # add output dir if it doesn't exist
    out_path = Path(PROCESSED_DOODLE_PATH)
    if not out_path.is_dir():
        out_path.mkdir(parents=True)
    # loop through doodles and preprocess
    for c in tqdm(categories):
        doodle_file_name = PROCESSED_DOODLE_PATH + c + '.npy'
        if Path(doodle_file_name).is_file():
            continue
        image = np.load(FULL_DOODLE_PATH + c + '.npy')
        full = []
        for i in range(len(image)):
            im = np.reshape(np.invert(image[i]), (28, 28))
            res = cv2.resize(im, dsize=(d_w, d_h), interpolation=cv2.INTER_CUBIC)
            v = np.reshape(res, (d_w * d_h,))
            full.append(v)
        np.save(doodle_file_name, full)
        doodle_files.append(doodle_file_name)
    return doodle_files

def process_images(categories, img_w, img_h, store_vgg):
    '''
    resize images into specified dims and save to PROCESSED_IMG_PATH_SM
    '''
    img_files = []
    vgg16 = models.vgg16(pretrained=True)
    torch.no_grad()
    # add output dir if it doesn't exist
    out_path = Path(PROCESSED_IMG_PATH_SM)
    if not out_path.is_dir():
        out_path.mkdir(parents=True)
    out_path = Path(VGG_IMG_FEATURES_PATH)
    if not out_path.is_dir():
        out_path.mkdir(parents=True)
    # loop through images and preprocess
    for c in tqdm(categories):
        vgg_file_name = VGG_IMG_FEATURES_PATH + c + '.npy'
        img_file_name = PROCESSED_IMG_PATH_SM + c + '.npy'
        if Path(vgg_file_name).is_file() and Path(img_file_name).is_file():
            continue
        full = []
        full_vgg = []
        for img_file in glob.glob(IMG_PATH + c + '/' + '*.jpg'):
            try:
                img_grey = Image.open(img_file).convert('L')
                img_resized = img_grey.resize((img_w, img_h), Image.ANTIALIAS)
                img_arr = np.array(img_resized, dtype=float)
                v = np.reshape(img_arr, (img_w * img_h,))
                full.append(v)
                # run through vgg16
                if store_vgg:
                    img = Image.open(img_file).convert('RGB')
                    img_resized = img.resize((img_w, img_h), Image.ANTIALIAS)
                    output = vgg16.features[:32](transforms.ToTensor()(img_resized).unsqueeze(0)).mean(-1).mean(-1).squeeze(0)
                    full_vgg.append(output)
            except:
                import traceback; traceback.print_exc()
        np.save(img_file_name, full)
        if store_vgg:
            np.save(vgg_file_name, full_vgg)
        img_files.append(img_file_name)
    return img_files

def create_features(categories, zero, num_labels, fname_suffix='train', store_vgg=True):
    '''
    randomly pair images with doodles and label based on matching or not
    '''
    aug = []
    img_dic = {}
    doodle_dic = {}
    true_labels = []
    false_labels = []
    X = []
    X_vgg = []
    y = []
    vgg_dic = {}
    for c in tqdm(categories):
        img = np.load(PROCESSED_IMG_PATH_SM + c + '.npy')
        img_dic[c] = img
        doodle = np.load(PROCESSED_DOODLE_PATH + c + '.npy')
        doodle_dic[c] = doodle
        if store_vgg:
            vgg_file_name = VGG_IMG_FEATURES_PATH + c + '.npy'
            vgg_dic[c] = np.load(vgg_file_name, allow_pickle=True)

    num_labels_per_example = num_labels/float(len(categories))
    # false labels: (vectorized img + doodle, -1/0)
    for img_key1 in tqdm(img_dic.keys()):
        img = img_dic[img_key1]
        if store_vgg:
            vgg_img = vgg_dic[img_key1]
        for img_key2 in img_dic.keys():
            if img_key1 != img_key2:
                doodle = doodle_dic[img_key2]
                r_n_labels = int(num_labels_per_example) if num_labels_per_example > 1 else np.random.binomial(size=1, n=1, p=num_labels_per_example)
                r_doodle = np.random.randint(len(doodle), size=r_n_labels)
                r_img = np.random.randint(len(img), size=r_n_labels)
                for r in range(len(r_img)):
                    X.append(np.concatenate((img[r_img[r]], doodle[r_doodle[r]]), axis=None))
                    if store_vgg:
                        X_vgg.append(np.concatenate((vgg_img[r_img[r]].detach(), doodle[r_doodle[r]]), axis=None))
                    if zero:  
                        y.append(0)
                    else: 
                        y.append(-1)

    # true labels: (vectorized img + doodle, 1) 
    for img_key in tqdm(img_dic.keys()):
        img = img_dic[img_key]
        if store_vgg:
            vgg_img = vgg_dic[img_key]
        doodle = doodle_dic[img_key]
        r_img = np.random.randint(len(img), size=num_labels)
        r_doodle = np.random.randint(len(doodle), size=num_labels)
        for j in range(len(r_img)):
            if store_vgg:
                X_vgg.append(np.concatenate((vgg_img[r_img[j]].detach(), doodle[r_doodle[j]]), axis=None))
            X.append(np.concatenate((img[r_img[j]], doodle[r_doodle[j]]), axis=None))
            y.append(1)
    # add output dir if it doesn't exist
    out_path = Path(BITMAP_FEATURES)
    if not out_path.is_dir():
        out_path.mkdir(parents=True)
    np.save(BITMAP_FEATURES + 'X_'+fname_suffix, X)
    np.save(BITMAP_FEATURES + 'y_'+fname_suffix, y)
    if store_vgg:
        np.save(BITMAP_FEATURES + 'X_vgg_'+fname_suffix, X_vgg)
    return X, y

# image size (img_w, img_h), doodle size (d_w, d_h), # of labels, -1/0 for y
# categories = {'apple', 'bear', 'tree'} list of strings
def main(categories, img_w, img_h, d_w, d_h, num_labels, zero, store_vgg=True):
    img_files = process_images(categories, img_w, img_h, store_vgg)
    doodle_files = process_doodles(categories, d_w, d_h, store_vgg)
    train_val_split_categories = test_train_split(
        classes=categories, 
        outfile=MATCHING_OUTPUT_PATH + 'train_val_split_baseline.npy',
        split=.2
    )
    X, y = create_features(train_val_split_categories['train'], zero, num_labels, fname_suffix='train', store_vgg=store_vgg)
    X, y = create_features(train_val_split_categories['test'], zero, num_labels, fname_suffix='val', store_vgg=store_vgg)

if __name__=='__main__':
    # main(test_train_split()['train'], 14, 14, 14, 14, 10, True)
    main(test_train_split(outfile=DATA_PATH+'test_train_split.npy')['train'], 28,28,28,28,10, True, store_vgg=False)
    # main(test_train_split(outfile=DATA_PATH+'test_train_split.npy')['train'], 224, 224, 28, 28, 10, True)
