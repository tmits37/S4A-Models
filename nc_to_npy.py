import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
from tqdm.contrib.concurrent import process_map
from functools import partial
import os
from tqdm import tqdm 
from multiprocessing import Pool

import xarray as xr
from pycocotools.coco import COCO
import netCDF4
import cv2


IMG_SIZE = 366
BANDS = {
    'B02': 10, 'B03': 10, 'B04': 10, 'B08': 10,
    'B05': 20, 'B07': 20, 'B06': 20, 'B8A': 20, 'B11': 20, 'B12': 20,
    'B01': 60, 'B09': 60, 'B10': 60
}

# Extract patches based on this band
REFERENCE_BAND = 'B02'


def get_medians(bandmod, netcdf, group_freq='1MS'):

    if bandmod == 'nrgb':
        bands = ['B02', 'B03', 'B04', 'B08']
        height, width = IMG_SIZE, IMG_SIZE
        expand_ratio = 1

    elif bandmod == 'rdeg':
        bands = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        height, width = int(IMG_SIZE//2), int(IMG_SIZE//2)
        expand_ratio = 2
    
    else:
        raise RuntimeError
    
    # Grab year from netcdf4's global attribute
    year = netcdf.patch_year
    date_range = pd.date_range(start=f'{year}-01-01', end=f'{int(year) + 1}-01-01', freq=group_freq)
    window = len(date_range) -1

    medians = np.empty((len(bands), window, height, width), dtype=np.uint16)

    for band_id, band in enumerate(bands):
        # Load band data
        band_data = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf[band]))
        band_data = band_data.sortby('time')

        # Aggregate into time bins
        band_data = band_data.groupby_bins(
            'time',
            bins=date_range,
            right=True,
            include_lowest=False,
            labels=date_range[:-1]
        ).median(dim='time')

        # Upsample so months without data are initiated with NaN values
        band_data = band_data.resample(time_bins=group_freq).median(dim='time_bins')
        
        # Fill:
        # NaN months with linear interpolation
        # NaN months outsize of range (e.x month 12) using extrapolation
        band_data = band_data.interpolate_na(dim='time_bins', method='linear', fill_value='extrapolate')

        # Keep values within requested time window
        band_data = band_data.isel(time_bins=slice(0, 0 + window))

        # Convert to numpy array
        band_data = band_data[f'{band}'].values

        # If resolution does not match reference band, stretch it
        # if expand_ratio != 1:
        #     band_data = cv2.resize(band_data.transpose(1,2,0), dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        #     band_data = band_data.transpose(2,0,1)

        medians[band_id, :, :, :] = np.expand_dims(band_data, axis=0)

    # Reshape so window length is first
    return medians.transpose(1, 0, 2, 3)   # (T, B, H, W)




def process_patch(out_path, mode,  data_path, bandmod, patch):
    patch_id, patch_info = patch # list(coco.imgs.items())
    filename = os.path.basename(patch_info['file_name']).replace('.nc', '.npy')

    netcdf = netCDF4.Dataset(os.path.join(data_path, patch_info['file_name']), 'r')
    medians = get_medians(bandmod, netcdf).astype(np.uint16)
    np.save(os.path.join(out_path, mode, bandmod, filename), medians)

    # Save labels
    labels = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf['labels']))['labels'].values
    # shape: (subpatches_in_row, subpatches_in_col, height, width)
    labels = labels.squeeze().astype(np.uint16)
    np.save(os.path.join(out_path, mode, 'label', filename), labels)

    return None

def map_function(data):
    process_patch(**data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and export median files for a given S2 dataset')
    parser.add_argument('--data', type=str, default='dataset', required=False,
                        help='Path to the netCDF files. Default "dataset/".')
    parser.add_argument('--root_coco_path', type=str, default='coco_files/', required=False,
                        help='Root path for coco file. Default "coco_files/".')
    parser.add_argument('--prefix_coco', type=str, default=None, required=False,
                        help='The prefix to use for the COCO file. Default none.')
    parser.add_argument('--out_path', type=str, default='logs/medians', required=False,
                        help='Path to export the medians into. Default "logs/medians/".')
    parser.add_argument('--output_size', nargs='+', default=None, required=False,
                        help='The size of the medians. If none given, the output will be of the same size.')
    parser.add_argument('--bands', nargs='+', default=None, required=False,
                        help='The bands to use. Default all.')
    parser.add_argument('--num_workers', type=int, default=8, required=False,
                        help='The number of workers to use for parallel computation. Default 8.')
    parser.add_argument('--band_mode', default='nrgb', choices=['nrgb', 'rdeg'])
    args = parser.parse_args()

    data_path = args.data
    out_path = args.out_path
    root_coco_path = args.root_coco_path

    medians_dtype = np.float32
    label_dtype = np.int16

    if args.bands is None:
        bands = BANDS.keys()
    else:
        bands = args.bands
    bands = sorted(bands)

    os.makedirs(out_path, exist_ok=True)
    num_buckets = len(pd.date_range(start=f'2020-01-01', end=f'2021-01-01', freq='1MS')) - 1

    bandmod = args.band_mode
    if bandmod == 'nrgb':
        bands = ['B02', 'B03', 'B04', 'B08']
        height, width = IMG_SIZE, IMG_SIZE

    elif bandmod == 'rdeg':
        bands = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        height, width = int(IMG_SIZE//2), int(IMG_SIZE//2)

    print(f'shape of output: ({num_buckets}, {len(bands)}, {height}, {height})')
    print('band_keys: ', bands)
    print(f'Saving into: {out_path}.')
    print(f'\nStart process...')

    for mode in ['train', 'val', 'test']:
        if args.prefix_coco is not None:
            coco_path = os.path.join(root_coco_path, f'{args.prefix_coco}_coco_{mode}.json')
        else:
            coco_path = os.path.join(root_coco_path, f'coco_{mode}.json')
        coco = COCO(coco_path)
        
        os.makedirs(os.path.join(out_path, mode), exist_ok=True)
        os.makedirs(os.path.join(out_path, mode, bandmod), exist_ok=True)
        os.makedirs(os.path.join(out_path, mode, 'label'), exist_ok=True)

        input_args = list()
        for coim in list(coco.imgs.items()):
            input_args.append(
                dict(out_path=out_path,
                     mode=mode,
                     data_path=data_path,
                     bandmod=bandmod,
                     patch=coim)
            )
        pool = Pool(args.num_workers)
        for _ in tqdm(pool.imap_unordered(map_function, input_args), total=len(list(coco.imgs.items()))):
            pass
        pool.close()
        pool.join()

    print('Dataset saved.\n')
    """
        python3 nc_to_npy.py --prefix_coco exp1_patches5000_strat --out_path dataset/newdata \
                             --band_mode nrgb --num_workers 20
    """
