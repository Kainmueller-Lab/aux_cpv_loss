import argparse
import os
import sys

import h5py
import mahotas
import numpy as np
import scipy.ndimage

import zarr


def watershed(surface, markers, fg, outFl, its=1):
    # compute watershed
    ws = mahotas.cwatershed(surface, markers)

    # write watershed directly
    print("watershed output: ", ws.shape, ws.dtype, ws.max(), ws.min())
    wsUI = ws.astype(np.uint16)
    outFl.create_dataset('volumes/watershed_seg', data = wsUI,
                         compression='gzip')

    # overlay fg and write
    wsFG = ws * fg
    print("watershed (foreground only):", wsFG.shape, wsFG.dtype,
          wsFG.max(), wsFG.min())
    wsFGUI = wsFG.astype(np.uint16)
    outFl.create_dataset('volumes/watershed_seg_fg', data = wsFGUI,
                         compression='gzip')
    for label in np.unique(wsFGUI):
        if label == 0:
            continue
        label_mask = wsFGUI==label
        dilated_label_mask = scipy.ndimage.binary_dilation(label_mask,
                                                           iterations=its)
        wsFGUI[dilated_label_mask] = label
    outFl.create_dataset('volumes/watershed_seg_fg_dilated', data = wsFGUI,
                         compression='gzip')


def label(**kwargs):
    sample = os.path.join(kwargs['pred_folder'],
                          kwargs['sample'] + "." + kwargs['pred_format'])
    if kwargs['pred_format'] == "zarr":
        input_file =  zarr.open(sample, 'r')
    elif kwargs['pred_format'] == "hdf":
        input_file =  h5py.File(sample, 'r')
    else:
        raise NotImplementedError("invalid pred format")
    affs = np.array(input_file[kwargs['aff_key']])
    fgbg = np.array(input_file[kwargs['fgbg_key']])
    raw = np.array(input_file[kwargs['raw_key']])

    if kwargs['pred_format'] == "hdf":
        input_file.close()

    if kwargs['output_format'] == "hdf":
        output_file = h5py.File(
            os.path.join(kwargs['output_folder'],
                         kwargs['sample'] + "."+kwargs['output_format']), 'w')
    else:
        raise NotImplementedError("invalid output format")

    # write fgbg prediction to file
    output_file.create_dataset('volumes/fgbg', data = fgbg,
                               compression='gzip')

    # threshold bg/fg
    fgbg = 1.0 *(fgbg > kwargs['fg_thresh'])

    # combine affinities
    affs_xyz = 1.0 - 0.33*(affs[0] + affs[1] + affs[2])
    output_file.create_dataset('volumes/affs', data = affs_xyz,
                               compression='gzip')

    output_file.create_dataset('volumes/raw', data = raw, compression='gzip')

    # load gt
    if 'gt' in kwargs and kwargs['gt'] is not None:
        if kwargs['gt_format'] == "hdf":
            with h5py.File(kwargs['gt'], 'r') as gt:
                gt_labels = np.array(gt[kwargs['gt_key']])
        elif kwargs['gt_format'] == "zarr":
            with zarr.open(kwargs['gt'], 'r') as gt:
                gt_labels = np.array(gt[kwargs['gt_key']])
        else:
            raise NotImplementedError("invalid gt format")
        print("gt min/max", gt_labels.min(), gt_labels.max())

    # compute markers for watershed (seeds)
    # TODO: check
    # markers = 1*(affs_xyz > kwargs['aff_thresh'])
    seeds = 1*(affs > kwargs['aff_thresh'])
    seeds = 0.33*(seeds[0] + seeds[1] + seeds[2])
    print("seeds min/max", np.min(seeds), np.max(seeds))
    seeds = seeds.astype('uint16')
    output_file.create_dataset('volumes/seeds', data = seeds,
                               compression='gzip')
    markers, cnt = scipy.ndimage.label(seeds)
    print("markers min/max, cnt", np.min(markers), np.max(markers), cnt)

    # compute watershed
    watershed(affs_xyz, markers, fgbg, output_file,
              its=kwargs['num_dilations'])

    if kwargs['output_format'] == "hdf":
        output_file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample', required=True)
    parser.add_argument('--pred_folder', required=True)
    parser.add_argument('--pred_format', default='zarr')
    parser.add_argument('--aff_key', default='volumes/pred_affs')
    parser.add_argument('--fgbg_key', default='volumes/pred_fgbg')
    parser.add_argument('--raw_key', default='volumes/raw_cropped')
    parser.add_argument('--gt')
    parser.add_argument('--gt_key', default='volumes/gt_labels')
    parser.add_argument('--gt_format', default='hdf')
    parser.add_argument('-o', '--output_folder', required=True)
    parser.add_argument('--output_format', default='hdf')
    parser.add_argument('-d', '--num_dilations', default=1, type=int)
    parser.add_argument('--fg_thresh', default=0.95, type=float)
    parser.add_argument('--aff_thresh', default=0.98, type=float)
    args = parser.parse_args()
    label(vars(args))

if __name__ == "__main__":
    main()
