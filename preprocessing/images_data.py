from pathlib import Path
import traceback
import numpy as np
import random
from tqdm import tqdm

from google_images_download import google_images_download

from util.constants import *

def download_images(n_imgs=15):
    '''
    Use google images api to download images and save to save_path
    '''
    # constants
    max_retries = 3
    # set up downloader
    ggl_img_resp = google_images_download.googleimagesdownload()
    doodle_dir = Path(FULL_DOODLE_PATH)
    assert(doodle_dir.is_dir())
    # make output directories if they don't exist
    out_path = Path(IMG_PATH)
    if not out_path.is_dir():
        out_path.mkdir(parents=True)
    # iterate over doodles we have and populate directories of images
    for f_doodle in tqdm(doodle_dir.iterdir()):
        doodle_class = f_doodle.stem
        img_class_path = out_path/doodle_class
        # check if images downloaded already -- if so, continue to next class
        downloaded_imgs = lambda: [i for i in img_class_path.rglob('*')]
        if len(downloaded_imgs()):
            continue
        # download images
        try:
            n_retries = 0
            while n_retries < max_retries and len(downloaded_imgs()) <= 0:
                ggl_img_resp.download({
                    'keywords': classname_to_keyword(doodle_class),
                    'limit': n_imgs,
                    'type': 'photo',
                    'output_directory': str(img_class_path),
                    'no_directory': True,
                    'silent_mode': True
                })
                n_retries += 1
        except:
            print('failed to download ' + doodle_class)
            traceback.print_exc()

def test_train_split(outfile='test_train_split.npy', split=40):
    '''
    randomly selects `split` fraction of classes to be reserved for test => outputs list of test 
    classes to `outfile` (placed inside preprocessing/data directory)
    '''
    outfile = DATA_PATH+outfile
    if Path(outfile).is_file():
        return np.load(outfile, allow_pickle=True).item()
    else:
        if split < 1:
            split *= N_IMG_CLASSES
        classes = [i.stem for i in Path(IMG_PATH).glob('*') if i != Path(PROCESSED_IMG_PATH)]
        random.shuffle(classes)
        classes_split = {
            'test': classes[:split],
            'train': classes[split:]
        }
        np.save(outfile, classes_split)
        return classes_split

def classname_to_keyword(name):
    '''
    replace underscores with spaces
    (files should have no spaces in name, but we want to search with normal words)
    '''
    return name.replace("_", " ")

if __name__=='__main__':
    download_images()
    test_train_split()