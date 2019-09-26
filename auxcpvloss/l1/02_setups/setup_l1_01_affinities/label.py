import argparse
import logging
import os
import sys

import h5py
import mahotas
import numpy as np
import scipy.ndimage

import zarr

logger = logging.getLogger(__name__)

def watershed(sample, surface, markers, fg, outFl, its=1):
    # compute watershed
    ws = mahotas.cwatershed(surface, markers)

    # write watershed directly
    logger.debug("%s: watershed output: %s %s %f %f",
                 sample, ws.shape, ws.dtype, ws.max(), ws.min())
    wsUI = ws.astype(np.uint16)
    outFl.create_dataset('volumes/watershed_seg', data=wsUI,
                         compression='gzip')

    # overlay fg and write
    wsFG = ws * fg
    logger.debug("%s: watershed (foreground only): %s %s %f %f",
                 sample, wsFG.shape, wsFG.dtype, wsFG.max(),
                 wsFG.min())
    wsFGUI = wsFG.astype(np.uint16)
    outFl.create_dataset('volumes/watershed_seg_fg', data=wsFGUI,
                         compression='gzip')
    for lbl in np.unique(wsFGUI):
        if lbl == 0:
            continue
        label_mask = wsFGUI == lbl
        dilated_label_mask = scipy.ndimage.binary_dilation(label_mask,
                                                           iterations=its)
        wsFGUI[dilated_label_mask] = lbl
    outFl.create_dataset('volumes/watershed_seg_fg_dilated', data=wsFGUI,
                         compression='gzip')


def label(**kwargs):
    logger.info("labelling %s %s", kwargs['sample'], kwargs['gt'])
    sample = os.path.join(kwargs['pred_folder'],
                          kwargs['sample'] + "." + kwargs['pred_format'])
    if kwargs['pred_format'] == "zarr":
        input_file = zarr.open(sample, 'r')
    elif kwargs['pred_format'] == "hdf":
        input_file = h5py.File(sample, 'r')
    else:
        raise NotImplementedError("invalid pred format")
    surf = np.array(input_file[kwargs['surf_key']])
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
    output_file.create_dataset('volumes/fgbg', data=fgbg,
                               compression='gzip')

    # threshold bg/fg
    fg = 1.0 * (fgbg > kwargs['fg_thresh'])
    if np.count_nonzero(fg) == 0:
        raise RuntimeError("{}: no foreground found".format(kwargs['sample']))

    if surf.shape[0] > 1 and len(surf.shape) == 4:
        # combine surface components
        surf_scalar = 1.0 - 0.33 * (surf[0] + surf[1] + surf[2])
    else:
        surf_scalar = 1.0 - surf
    output_file.create_dataset('volumes/surf', data=surf_scalar,
                               compression='gzip')

    output_file.create_dataset('volumes/raw', data=raw, compression='gzip')

    # load gt
    if 'gt' in kwargs and kwargs['gt'] is not None:
        if kwargs['gt_format'] == "hdf":
            with h5py.File(kwargs['gt'], 'r') as gt:
                gt_labels = np.array(gt[kwargs['gt_key']])
        elif kwargs['gt_format'] == "zarr":
            gt = zarr.open(kwargs['gt'], 'r')
            gt_labels = np.array(gt[kwargs['gt_key']])
        else:
            raise NotImplementedError("invalid gt format")
        logger.debug("%s: gt min %f, max %f",
                     kwargs['sample'], gt_labels.min(), gt_labels.max())

    # compute markers for watershed (seeds)
    seeds = (1 * (surf > kwargs['seed_thresh'])).astype(np.uint8)
    if surf.shape[0] > 1 and len(surf.shape) == 4:
        seeds = (seeds[0] + seeds[1] + seeds[2])
        seeds = (seeds > 2).astype(np.uint8)
    logger.debug("%s: seeds min/max %f %f",
                 kwargs['sample'], np.min(seeds), np.max(seeds))

    if np.count_nonzero(seeds) == 0:
        logger.warning("%s: no seed points found for watershed", sample)

    output_file.create_dataset('volumes/seeds', data=seeds,
                               compression='gzip')
    markers, cnt = scipy.ndimage.label(seeds)
    logger.debug("%s: markers min %f, max %f, cnt %f",
                 kwargs['sample'], np.min(markers), np.max(markers), cnt)

    # compute watershed
    watershed(kwargs['sample'], surf_scalar, markers, fg, output_file,
              its=kwargs['num_dilations'])

    if kwargs['output_format'] == "hdf":
        output_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample', required=True)
    parser.add_argument('--pred_folder', required=True)
    parser.add_argument('--pred_format', default='zarr')
    parser.add_argument('--surf_key', default='volumes/pred_affs')
    parser.add_argument('--fgbg_key', default='volumes/pred_fgbg')
    parser.add_argument('--raw_key', default='volumes/raw_cropped')
    parser.add_argument('--gt')
    parser.add_argument('--gt_key', default='volumes/gt_labels')
    parser.add_argument('--gt_format', default='hdf')
    parser.add_argument('-o', '--output_folder', required=True)
    parser.add_argument('--output_format', default='hdf')
    parser.add_argument('-d', '--num_dilations', default=1, type=int)
    parser.add_argument('--fg_thresh', default=0.95, type=float)
    parser.add_argument('--surf_thresh', default=0.98, type=float)
    args = parser.parse_args()
    label(**vars(args))

if __name__ == "__main__":
    main()
