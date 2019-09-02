import logging
import glob
import os

import h5py
import numpy as np
import pymrt as mrt
import pymrt.geometry
import scipy.spatial
import toml
import zarr

logger = logging.getLogger(__name__)


def evaluate_file(**kwargs):
    logger.info("evaluating %s %s", kwargs['res_file'], kwargs['gt_file'])
    sample_pred = os.path.basename(kwargs['res_file'])

    # load labelling
    if kwargs['res_file'].endswith("zarr"):
        input_file = zarr.open(kwargs['res_file'], 'r')
    elif kwargs['res_file'].endswith("hdf"):
        input_file = h5py.File(kwargs['res_file'], 'r')
    else:
        raise NotImplementedError("invalid pred format")
    raw = np.array(input_file['/volumes/raw_cropped'])
    reg_max = np.squeeze(np.array(input_file['/volumes/markers']), axis=0)
    pred_cells = np.argwhere(reg_max > 0)
    logger.info("%s: number pred cells: %s", sample_pred, pred_cells.shape)

    # load gt image
    sample_gt = os.path.basename(kwargs['gt_file'])
    if kwargs['gt_file'].endswith("hdf"):
        with h5py.File(kwargs['gt_file'], 'r') as gt:
            gt_labels = np.array(gt[kwargs['gt_key']])
    elif kwargs['gt_file'].endswith("zarr"):
        gt = zarr.open(kwargs['gt_file'], 'r')
        gt_labels = np.array(gt[kwargs['gt_key']])
    else:
        raise NotImplementedError("invalid gt format")
    gt_labels = np.squeeze(gt_labels, axis=0)
    logger.debug("%s: gt min %f, max %f",
                 sample_gt, gt_labels.min(), gt_labels.max())

    # load gt points
    gt_cells = []
    csvFn = os.path.splitext(kwargs['gt_file'])[0] + ".csv"
    with open(csvFn, 'r') as f:
        for ln in f:
            z, y, x, _ = ln.strip().split(",")
            z = float(z)-kwargs.get("padding", [0, 0, 0])[0]
            y = float(y)-kwargs.get("padding", [0, 0, 0])[1]
            x = float(x)-kwargs.get("padding", [0, 0, 0])[2]
            gt_cells.append((z,y,x))
    gt_cells = np.array(gt_cells)
    logger.info("%s: number gt cells: %s", sample_gt, gt_cells.shape)
    if kwargs['debug']:
        gt_labels_debug = np.array(input_file['/volumes/gt_labels_debug'])
    else:
        gt_labels_debug = None


    outFnBase = os.path.join(
        kwargs['out_dir'],
        os.path.splitext(os.path.basename(kwargs['res_file']))[0] +
        kwargs['res_key'].replace("/","_") + kwargs['suffix'] + "_scores")
    if len(glob.glob(outFnBase + "*")) > 0:
        logger.info('Skipping evaluation for %s. Already exists!',
                    kwargs['res_file'])
        tomlFl = open(outFnBase+".toml", 'r')
        return toml.load(tomlFl)

    tomlFl = open(outFnBase + ".toml", 'w')
    results = {}
    res = computeMetrics(raw, gt_cells, pred_cells,
                         gt_labels, gt_labels_debug,
                         draw=kwargs['debug'], **kwargs)
    results['gt_pred'] = res
    res = computeMetrics(raw, pred_cells, gt_cells,
                         gt_labels, gt_labels_debug,
                         draw=False, reverse=True, **kwargs)
    results['pred_gt'] = res
    toml.dump(results, tomlFl)
    return results


def computeMetrics(raw, source_cells, target_cells,
                   gt_labels, gt_labels_debug,
                   draw=False, reverse=False, **kwargs):
    source_cells_tree = scipy.spatial.cKDTree(source_cells, leafsize=4)

    nn_distances, nn_locations = source_cells_tree.query(target_cells)

    fpP = 0
    fnGT = 0
    tpP = 0
    tpGT = 0
    nsP = 0
    fsGT = 0
    fn = 0
    # gt_labels_matched_num = {}


    cntsSource = np.zeros(source_cells.shape[0], dtype=np.uint16)
    cntsTarget = np.zeros(target_cells.shape[0], dtype=np.uint16)

    for dist, sID, tID in zip(nn_distances, nn_locations,
                               range(target_cells.shape[0])):
        if kwargs['debug'] and tID >= 20:
            break
        logger.debug("checking nearest neighbor target cell: %s", tID)
        if sID < source_cells.shape[0]:
            logger.debug("%s %s %s", dist, source_cells[sID],
                         target_cells[tID])
            # if within distance and same label
            if dist < kwargs['distance_limit'] and \
               gt_labels[int(round(target_cells[tID][0])),
                         int(round(target_cells[tID][1])),
                         int(round(target_cells[tID][2]))] == \
               gt_labels[int(round(source_cells[sID][0])),
                         int(round(source_cells[sID][1])),
                         int(round(source_cells[sID][2]))]:
                l = gt_labels[int(round(target_cells[tID][0])),
                              int(round(target_cells[tID][1])),
                              int(round(target_cells[tID][2]))]
                # TODO: check
                # if l in gt_labels_matched_num:
                #     gt_labels_matched_num[l] += 1
                # else:
                #     gt_labels_matched_num[l] = 1
                cntsSource[sID] += 1
                cntsTarget[tID] += 1
                # colID = 0
            else:
                fpP += 1
                # colID = 1


        else:
            logger.debug("no neighbor for %s", sID)
            fpP += 1

    results = {}
    if reverse:
        tpP = np.count_nonzero(cntsSource==1)
        fpP = np.count_nonzero(cntsSource==0)
        # non-split
        nsP = np.count_nonzero(cntsSource>1)

        tpGT = np.count_nonzero(cntsTarget==1)
        fnGT = np.count_nonzero(cntsTarget==0)

        results['Num_GT'] = target_cells.shape[0]
        results['Num_Pred'] = source_cells.shape[0]
    else:
        fnGT = np.count_nonzero(cntsSource==0)
        tpGT = np.count_nonzero(cntsSource==1)
        # false-split
        fsGT = np.count_nonzero(cntsSource>1)

        tpP = np.count_nonzero(cntsTarget==1)
        fpP = np.count_nonzero(cntsTarget!=1)

        results['Num_GT'] = source_cells.shape[0]
        results['Num_Pred'] = target_cells.shape[0]

    logger.debug("Num GT: %s", results['Num_GT'])
    logger.debug("Num Pred: %s", results['Num_Pred'])
    results['TP_GT'] = tpGT
    logger.debug("TP GT: %s", results['TP_GT'])
    results['FN_GT'] = fnGT
    logger.debug("FN GT: %s", results['FN_GT'])
    results['FS_GT'] = fsGT
    logger.debug("FS GT: %s", results['FS_GT'])
    results['TP_Pred'] = tpP
    logger.debug("TP Pred: %s", results['TP_Pred'])
    results['FP_Pred'] = fpP
    logger.debug("FP Pred: %s", results['FP_Pred'])
    results['NS_Pred'] = nsP
    logger.debug("NS Pred: %s", results['NS_Pred'])

    apDef = tpP / (tpP + fnGT + fpP)
    results['AP'] = apDef
    logger.debug("AP: %s", results['AP'])
    apSD = tpP / (tpP+fpP)
    results['AP_CV'] = apSD
    logger.debug("AP_CV: %s", results['AP_CV'])

    return results
