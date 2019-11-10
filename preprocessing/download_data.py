from google_images_download import google_images_download
from pathlib import Path
import traceback

from util.data_paths import *

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
    for f_doodle in doodle_dir.iterdir():
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
                    'keywords': doodle_class,
                    'limit': n_imgs,
                    'type': 'photo',
                    'output_directory': str(img_class_path),
                    'no_directory': True
                })
                n_retries += 1
        except:
            print('failed to download ' + doodle_class)
            traceback.print_exc()

def download_doodles(save_path):
    '''
    (going to attempt to do this automatically rather than having to download each set of doodles separately)
    '''
    pass

def test_val_split(imgs_dir, outfile, split=.2):
    '''
    randomly selects `split` fraction of classes to be reserved for test => outputs list of test 
    classes to `outfile`
    '''
    pass

if __name__=='__main__':
    download_images()