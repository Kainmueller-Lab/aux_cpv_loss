import argparse
import logging
import os
import sys

import h5py
import mahotas
import numpy as np
import scipy.ndimage

import waterz
import zarr

logger = logging.getLogger(__name__)

def watershed(sample, surface, markers, fg, its=1):
    # compute watershed
    ws = mahotas.cwatershed(surface, markers)

    # write watershed directly
    logger.debug("%s: watershed output: %s %s %f %f",
                 sample, ws.shape, ws.dtype, ws.max(), ws.min())
    wsUI = ws.astype(np.uint16)

    # overlay fg and write
    wsFG = ws * fg
    logger.debug("%s: watershed (foreground only): %s %s %f %f",
                 sample, wsFG.shape, wsFG.dtype, wsFG.max(),
                 wsFG.min())
    wsFGUI = wsFG.astype(np.uint16)

    wsFGUIdil = np.copy(wsFGUI)
    if its == 0:
        return wsUI, wsFGUI, wsFGUIdil

    for lbl in np.unique(wsFGUIdil):
        if lbl == 0:
            continue
        label_mask = wsFGUIdil == lbl
        dilated_label_mask = scipy.ndimage.binary_dilation(label_mask,
                                                           iterations=its)
        wsFGUIdil[dilated_label_mask] = lbl

    return wsUI, wsFGUI, wsFGUIdil


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

    if kwargs.get('waterz'):
        affinities = np.array(input_file[kwargs['surf_key']])
        segmentations = list(waterz.agglomerate(affinities, [kwargs['waterz_threshold']]))[0].astype(np.uint32)
        print(segmentations.shape, segmentations.dtype, len(np.unique(segmentations)), kwargs['waterz_threshold'])
        segmentations_dil = np.copy(segmentations)
        if kwargs['num_dilations'] > 0 and len(np.unique(segmentations)) < 1000:
            segmentations_dil = np.copy(segmentations)
            for lbl in np.unique(segmentations_dil):
                if lbl == 0:
                    continue
                label_mask = segmentations_dil == lbl
                dilated_label_mask = scipy.ndimage.binary_dilation(label_mask,
                                                                   iterations=kwargs['num_dilations'])
                segmentations_dil[dilated_label_mask] = lbl
        if kwargs['output_format'] == "hdf":
            out_fn = os.path.join(kwargs['output_folder'],
                                  kwargs['sample'] + "."+kwargs['output_format'])
            with h5py.File(out_fn, 'w') as output_file:
                # write fgbg prediction to file
                output_file.create_dataset('volumes/watershed_seg_fg', data=segmentations,
                                           compression='gzip')
                output_file.create_dataset('volumes/watershed_seg_fg_dilated',
                                           data=segmentations_dil,
                                           compression='gzip')
        return
    surf = np.array(input_file[kwargs['surf_key']])
    fgbg = np.array(input_file[kwargs['fgbg_key']])
    raw = np.array(input_file[kwargs['raw_key']])

    if kwargs['pred_format'] == "hdf":
        input_file.close()

    # threshold bg/fg
    fg = 1.0 * (fgbg > kwargs['fg_thresh'])
    if np.count_nonzero(fg) == 0:
        logger.warning("{}: no foreground found".format(kwargs['sample']))

    # combine surface components
    surf_scalar = 1.0 - 0.33 * (surf[0] + surf[1] + surf[2])

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
    seeds = (seeds[0] + seeds[1] + seeds[2])
    seeds = (seeds > 2).astype(np.uint8)
    logger.info("%s: seeds min/max %f %f",
                kwargs['sample'], np.min(seeds), np.max(seeds))

    if np.count_nonzero(seeds) == 0:
        logger.warning("%s: no seed points found for watershed", sample)

    markers, cnt = scipy.ndimage.label(seeds)
    logger.debug("%s: markers min %f, max %f, cnt %f",
                 kwargs['sample'], np.min(markers), np.max(markers), cnt)

    # compute watershed
    wsUI, wsFGUI, wsFGUIdil = watershed(kwargs['sample'], surf_scalar,
                                        markers, fg,
                                        its=kwargs['num_dilations'])

    if kwargs['output_format'] == "hdf":
        out_fn = os.path.join(kwargs['output_folder'],
                              kwargs['sample'] + "."+kwargs['output_format'])
        with h5py.File(out_fn, 'w') as output_file:
            # write fgbg prediction to file
            output_file.create_dataset('volumes/fgbg', data=fgbg,
                                       compression='gzip')
            output_file.create_dataset('volumes/surf', data=surf_scalar,
                                       compression='gzip')
            output_file.create_dataset('volumes/raw', data=raw,
                                       compression='gzip')
            output_file.create_dataset('volumes/seeds', data=seeds,
                                       compression='gzip')
            output_file.create_dataset('volumes/watershed_seg', data=wsUI,
                         compression='gzip')
            output_file.create_dataset('volumes/watershed_seg_fg', data=wsFGUI,
                         compression='gzip')
            output_file.create_dataset('volumes/watershed_seg_fg_dilated',
                                 data=wsFGUIdil,
                                 compression='gzip')
    else:
        raise NotImplementedError("invalid output format")
