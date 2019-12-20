# Supervised Forest Classification with Sentinel-1 data

This code is first try to use Sentinel-1 data with labels from Global Forest Watch (GFW) to train a U-Net architecture for `forest` and `no forest` classification. 

The input data are reprojected using `re-project.ipynb` on to the same grid. 

The `pre-process.ipynb` notebook generates the 256 x 256 image chips. 

And the `main.ipynb` notebook runs the training and testing. 