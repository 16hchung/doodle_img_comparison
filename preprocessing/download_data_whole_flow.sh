############
# need to install gsutil, see: https://cloud.google.com/sdk/docs/
############

doodle_dir=preprocessing/data/doodles/full_numpy_bitmap/
n_classes=345

# download doodles from quick draw set
if ($(ls -l $doodle_dir | wc -l) < $n_classes); then
    gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/* $doodle_dir

    # rename files to no longer have spaces
    for file in $doodle_dir*
    do
      mv "$file" "${file// /_}"
    done
else
    echo "skipping doodle download"
fi

# download google images based on contents of $doodle_dir
python3 -m preprocessing.images_data