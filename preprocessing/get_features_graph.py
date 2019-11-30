import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
import glob
import pickle

from util.constants import *
from .images_data import test_train_split

def load_data_for_matcher(n_categories, n_doodles_per, n_images_per):
    # get random categories & save what they are
    categories = test_train_split(outfile=DATA_PATH+'test_train_split.npy')['train']
    categories = shuffle(categories)[:n_categories]
    # get random (count `n_doodles_per`) doodle vgg outputs per category
    # and get random (count `n_images_per`) image vgg outputs per category
    doodles_names = []
    doodle_vggs = []
    images_names = []
    image_vggs = []
    print('loading data')
    for c in tqdm(categories):
        vgg_file_name = VGG_IMG_FEATURES_PATH + c + '.npy'
        images_all = np.load(vgg_file_name, allow_pickle=True)
        vgg_doodle_file_name = VGG_DOODLE_FEATURES_PATH + c + '.npy'
        doodle_all = np.load(vgg_doodle_file_name, allow_pickle=True)

        r_img = np.random.randint(len(images_all), size=n_images_per)
        r_doodle = np.random.randint(len(doodle_all), size=n_doodles_per)
        doodles_names = doodles_names + [(c, r) for r in r_doodle]
        images_names = images_names + [(c, r) for r in r_img]
        doodle_vggs = doodle_vggs + [doodle_all[r] for r in r_doodle]
        image_vggs = image_vggs + [images_all[r] for r in r_img]
    return doodles_names, doodle_vggs, images_names, image_vggs

def get_clustering_features(n_categories, n_doodles_per, n_images_per, matcher, scaler, out_prefix=''):
    doodles_names, doodle_vggs, images_names, image_vggs = load_data_for_matcher(n_categories, n_doodles_per, n_images_per)
    # generate graph representation
    graph = []
    print('iterating over doodles, then images')
    for doodle_vgg in tqdm(doodle_vggs):
        pairs = []
        for image_vgg in image_vggs:
            pairs.append(np.concatenate((image_vgg.detach(),doodle_vgg.detach())))
        # run through model
        pairs = scaler.transform(pairs)
        matching = matcher.predict(pairs)
        doodle_row = [i for i,m in enumerate(matching) if m>0]
        graph.append(doodle_row)
    saved_info = {
        'X':graph,
        'doodle_rows':doodles_names,
        'image_cols':images_names
    }
    # save to disk
    out_suffix = '_{}d_{}i_{}cat_clustering.py'.format(n_doodles_per, n_images_per, n_categories)
    np.save(GRAPH_FEATURES + out_prefix + out_suffix, saved_info)

def get_stable_marriage_features(n_categories, n_doodles_per, n_images_per, matcher, scaler, out_prefix=''):
    doodles_names, doodle_vggs, images_names, image_vggs = load_data_for_matcher(n_categories, n_doodles_per, n_images_per)
    def generate_prefs(outside_names, outside_vggs, inside_names, inside_vggs, img_is_outside):
        prefs_rows = []
        for o_i, o_name in tqdm(enumerate(outside_names)):
            row = {
                'name': str(o_name),
                "is_free": True,
                "engaged_to": "",
                "proposed_to":[]
            }
            pairs = []
            for inside_vgg in inside_vggs:
                doodle_vgg = inside_vgg      if img_is_outside else doodle_vggs[o_i] 
                image_vgg  = image_vggs[o_i] if img_is_outside else inside_vgg
                pairs.append(np.concatenate((image_vgg.detach(),doodle_vgg.detach())))
            # run through model
            pairs = scaler.transform(pairs)
            matching = matcher.predict(pairs)
            sorted_img_idcs = np.argsort(matching)
            row['preferences'] = [str(inside_names[i]) for i in np.flip(sorted_img_idcs)]
            prefs_rows.append(row)
        return prefs_rows
    # arbitrarily set doodles to be males and images to be females
    print('generating males list')
    male_doodles = generate_prefs(doodles_names, doodle_vggs, images_names, image_vggs, False)
    print('generating female list')
    female_images = generate_prefs(images_names, image_vggs, doodles_names, doodle_vggs, True)
    saved_info = {'male (doodle)': male_doodles, 'female (image)': female_images}
    # save to disk
    out_suffix = '_{}d_{}i_{}cat_marriage.py'.format(n_doodles_per, n_images_per, n_categories)
    np.save(GRAPH_FEATURES + out_prefix + out_suffix, saved_info)

def load_object(object_path):
    with open(object_path, 'rb') as f:
        obj = pickle.load(f)
        return obj

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_fname', default='K_SVM_baseline_Cis10.00_accuris0.7752.pkl')
    parser.add_argument('--scaler_fname', default='moredata_scaler.pkl')
    parser.add_argument('--n_categories', type=int, default=5)
    parser.add_argument('--n_doodles_per', type=int, default=1)
    parser.add_argument('--n_images_per', type=int, default=1)
    parser.add_argument('--feature_type', default='pivot')
    parser.add_argument('--outfname_prefix', default='')
    args = parser.parse_args()

    model_path = MATCHING_OUTPUT_PATH + args.model_fname
    scaler_path = MATCHING_OUTPUT_PATH + args.scaler_fname
    clf = load_object(model_path)
    scaler = load_object(scaler_path)
    if args.feature_type == 'pivot':
        get_clustering_features(args.n_categories, args.n_doodles_per, args.n_images_per, clf, scaler, args.outfname_prefix)
    elif args.feature_type == 'marriage':
        get_stable_marriage_features(args.n_categories, args.n_doodles_per, args.n_images_per, clf, scaler, args.outfname_prefix)
    else:
        print('can\'t recognize feature type')