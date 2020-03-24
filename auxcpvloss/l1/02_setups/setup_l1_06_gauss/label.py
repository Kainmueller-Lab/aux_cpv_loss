import logging
import os

import h5py
import mahotas
import numpy as np
import zarr
import skimage.feature
import scipy.ndimage
logger = logging.getLogger(__name__)


def label(**kwargs):
    logger.info("labelling %s %s", kwargs['sample'], kwargs['gt'])

    # load prediction
    sample = os.path.join(kwargs['pred_folder'],
                          kwargs['sample'] + "." + kwargs['pred_format'])
    if kwargs['pred_format'] == "zarr":
        input_file = zarr.open(sample, 'r')
    elif kwargs['pred_format'] == "hdf":
        input_file = h5py.File(sample, 'r')
    else:
        raise NotImplementedError("invalid pred format")
    gauss = np.array(input_file['/volumes/pred_cp'])
    raw = np.array(input_file['/volumes/raw_cropped'])
    logger.debug("prediction min/max: %s %s", np.min(gauss), np.max(gauss))
    logger.debug("prediction shape %s", gauss.shape)
    logger.debug("raw shape %s", raw.shape)

    # # load gt image
    # if kwargs['gt_format'] == "hdf":
    #     with h5py.File(kwargs['gt'], 'r') as gt:
    #         gt_labels = np.array(gt[kwargs['gt_key']])
    # elif kwargs['gt_format'] == "zarr":
    #     gt = zarr.open(kwargs['gt'], 'r')
    #     gt_labels = np.array(gt[kwargs['gt_key']])
    # else:
    #     raise NotImplementedError("invalid gt format")
    # gt_labels = np.squeeze(gt_labels, axis=0)
    # logger.debug("%s: gt min %f, max %f",
    #              kwargs['sample'], gt_labels.min(), gt_labels.max())


    # markers = 1*(gauss > kwargs['gauss_thresh'])
    # logger.debug("pixel over threshold: %s (%s)", np.argwhere(markers).shape,
    #              markers.dtype)
    # if np.argwhere(markers).shape[0] == 0:
    #     logger.error("no blobs found for %s (0 pixel over threshold %s)",
    #                  kwargs['sample'], kwargs['gauss_thresh'])
    #     return

    # if kwargs['debug']:
    #     output_file.create_dataset('volumes/gauss', data=gauss,
    #                                dtype=np.float32, compression='gzip')


    # if kwargs['debug']:
    #     output_file.create_dataset('volumes/markers1',
    #                                data=markers.astype(np.uint16),
    #                                compression='gzip')

    # reg_max = 1*mahotas.regmax(gauss)
    # if kwargs['debug']:
    #     output_file.create_dataset('volumes/markers2',
    #                                data=reg_max.astype(np.uint16),
    #                                compression='gzip')
    # logger.debug("regional maxima: %s (%s)", np.argwhere(reg_max > 0).shape,
    #              markers.dtype)

    # reg_max = markers * reg_max
    # logger.debug("regional maxima over threshold: %s",
    #              np.argwhere(reg_max > 0).shape)
    # reg_max = reg_max.astype(np.uint16)
    # # TODO: check
    # gaussTmp = gauss.copy()
    # gaussTmp[gauss < kwargs['gauss_thresh']] = 0
    # gaussTmp = 1*mahotas.regmax(gaussTmp)
    # reg_max = gaussTmp

    gaussTmp = gauss.copy()
    min_distance = kwargs['nms_size']
    size = 2 * min_distance + 1
    image_max = scipy.ndimage.maximum_filter(gaussTmp, size=size, mode='constant')
    mask = gaussTmp == image_max
    mask &= gaussTmp > kwargs['gauss_thresh']
    print(np.count_nonzero(mask))
    reg_max = mask


    # create output file
    if kwargs['output_format'] == "hdf":
        output_file = h5py.File(
            os.path.join(kwargs['output_folder'],
                         kwargs['sample'] + "."+kwargs['output_format']), 'w')
    else:
        raise NotImplementedError("invalid output format")

    output_file.create_dataset('volumes/markers', data=reg_max,
                               dtype=np.float32, compression='gzip')
    output_file.create_dataset('volumes/pred_cp', data=gauss,
                               dtype=np.float32, compression='gzip')
    output_file.create_dataset('volumes/raw_cropped', data=raw,
                               compression='gzip')

    pred_cells = np.argwhere(reg_max > 0)
    logger.info("number predicted cells: %s", pred_cells.shape)


    if kwargs['debug']:
        gt_labels_debug = 1*(gt_labels>0)
        gt_labels_debug = gt_labels_debug.astype(np.float32)
        gt_labels_debug = 0.5*gt_labels_debug
        gt_labels_debug = np.stack((gt_labels_debug,
                                    gt_labels_debug,
                                    gt_labels_debug))
        logger.debug("gt labels debug shape: %s", gt_labels_debug.shape)
        output_file.create_dataset('volumes/gt_labels_debug',
                                   data=gt_labels_debug,
                                   compression='gzip')
