from __future__ import print_function
import json
import logging
import os
import sys
import time

import h5py
import numpy as np
import tensorflow as tf
import zarr

import gunpowder as gp

logger = logging.getLogger(__name__)

class NoOp(gp.BatchFilter):

    def __init__(self):
        pass

    def process(self, batch, request):
        pass


def train_until(**kwargs):
    if tf.train.latest_checkpoint(kwargs['output_folder']):
        trained_until = int(
            tf.train.latest_checkpoint(kwargs['output_folder']).split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= kwargs['max_iteration']:
        return

    anchor = gp.ArrayKey('ANCHOR')
    raw = gp.ArrayKey('RAW')
    raw_cropped = gp.ArrayKey('RAW_CROPPED')
    gt_sdt = gp.ArrayKey('GT_SDT')

    pred_sdt = gp.ArrayKey('PRED_SDT')

    pred_sdt_gradients = gp.ArrayKey('PRED_SDT_GRADIENTS')


    with open(os.path.join(kwargs['output_folder'],
                           kwargs['name'] + '_config.json'), 'r') as f:
        net_config = json.load(f)
    with open(os.path.join(kwargs['output_folder'],
                           kwargs['name']  + '_names.json'), 'r') as f:
        net_names = json.load(f)

    voxel_size = gp.Coordinate(kwargs['voxel_size'])
    input_shape_world = gp.Coordinate(net_config['input_shape'])*voxel_size
    output_shape_world = gp.Coordinate(net_config['output_shape'])*voxel_size

    # formulate the request for what a batch should (at least) contain
    request = gp.BatchRequest()
    request.add(raw, input_shape_world)
    request.add(raw_cropped, output_shape_world)
    request.add(gt_sdt, output_shape_world)
    request.add(anchor, output_shape_world)

    # when we make a snapshot for inspection (see below), we also want to
    # request the predicted affinities and gradients of the loss wrt the
    # affinities
    snapshot_request = gp.BatchRequest()
    snapshot_request.add(raw_cropped, output_shape_world)
    snapshot_request.add(gt_sdt, output_shape_world)
    snapshot_request.add(pred_sdt, output_shape_world)
    snapshot_request.add(pred_sdt_gradients, output_shape_world)


    if kwargs['input_format'] != "hdf" and kwargs['input_format'] != "zarr":
        raise NotImplementedError("train node for %s not implemented yet",
                                  kwargs['input_format'])

    fls = []
    for f in kwargs['data_files']:
        fls.append(os.path.splitext(f)[0])
    ln = len(fls)
    print("first 5 files: ", fls[0:4])

    # padR = 46
    # padGT = 32

    if kwargs['input_format'] == "hdf":
        sourceNode = gp.Hdf5Source
    elif kwargs['input_format'] == "zarr":
        sourceNode = gp.ZarrSource

    augmentation = kwargs['augmentation']
    pipeline = (
        tuple(
            sourceNode(
                fls[t] + "." + kwargs['input_format'],
                datasets={
                    raw: 'volumes/raw',
                    gt_sdt: 'volumes/gt_tanh',
                    anchor: 'volumes/gt_tanh',
                },
                array_specs={
                    raw: gp.ArraySpec(interpolatable=True),
                    gt_sdt: gp.ArraySpec(interpolatable=False),
                    anchor: gp.ArraySpec(interpolatable=False)
                }
            )
            + gp.Pad(raw, None)
            + gp.Pad(gt_sdt, None)

            # chose a random location for each requested batch
            + gp.RandomLocation()

            for t in range(ln)
        ) +

        # chose a random source (i.e., sample) from the above
        gp.RandomProvider() +

        # elastically deform the batch
        (gp.ElasticAugment(
            augmentation['elastic']['control_point_spacing'],
            augmentation['elastic']['jitter_sigma'],
            [augmentation['elastic']['rotation_min']*np.pi/180.0,
             augmentation['elastic']['rotation_max']*np.pi/180.0],
            subsample=augmentation['elastic'].get('subsample', 1)) \
        if augmentation.get('elastic') is not None else NoOp())  +

        # apply transpose and mirror augmentations
        gp.SimpleAugment(mirror_only=augmentation['simple'].get("mirror"),
                         transpose_only=augmentation['simple'].get("transpose")) +

        # # scale and shift the intensity of the raw array
        # # scale and shift the intensity of the raw array
        (gp.IntensityAugment(
            raw,
            scale_min=augmentation['intensity']['scale'][0],
            scale_max=augmentation['intensity']['scale'][1],
            shift_min=augmentation['intensity']['shift'][0],
            shift_max=augmentation['intensity']['shift'][1],
            z_section_wise=False) \
        if augmentation.get('intensity') is not None else NoOp()) +

        (gp.IntensityScaleShift(
            raw,
            scale=augmentation['scale_shift']['scale'],
            shift=augmentation['scale_shift']['shift']) \
         if augmentation.get('scale_shift') is not None else NoOp()) +

        # pre-cache batches from the point upstream
        gp.PreCache(
            cache_size=kwargs['cache_size'],
            num_workers=kwargs['num_workers']) +

        # perform one training iteration for each passing batch (here we use
        # the tensor names earlier stored in train_net.config)
        gp.tensorflow.Train(
            os.path.join(kwargs['output_folder'], kwargs['name']),
            optimizer=net_names['optimizer'],
            summary=net_names['summaries'],
            log_dir=kwargs['output_folder'],
            loss=net_names['loss'],
            inputs={
                net_names['raw']: raw,
                net_names['gt_sdt']: gt_sdt,
                net_names['anchor']: anchor,
            },
            outputs={
                net_names['pred_sdt']: pred_sdt,
                net_names['raw_cropped']: raw_cropped,
            },
            gradients={
                net_names['pred_sdt']: pred_sdt_gradients,
            },
            save_every=kwargs['checkpoints']) +

        # save the passing batch as an HDF5 file for inspection
        gp.Snapshot(
            {
                raw: '/volumes/raw',
                raw_cropped: 'volumes/raw_cropped',
                gt_sdt: '/volumes/gt_sdt',
                pred_sdt: '/volumes/pred_sdt',
                pred_sdt_gradients: '/volumes/pred_sdt_gradients',
            },
            output_dir=os.path.join(kwargs['output_folder'], 'snapshots'),
            output_filename='batch_{iteration}.hdf',
            every=kwargs['snapshots'],
            additional_request=snapshot_request,
            compression_type='gzip') +

        # show a summary of time spend in each node every 10 iterations
        gp.PrintProfilingStats(every=kwargs['profiling'])
    )

    #########
    # TRAIN #
    #########
    print("Starting training...")
    with gp.build(pipeline):
        print(pipeline)
        for i in range(trained_until, kwargs['max_iteration']):
            # print("request", request)
            start = time.time()
            pipeline.request_batch(request)
            time_of_iteration = time.time() - start

            logger.info(
                "Batch: iteration=%d, time=%f",
                i, time_of_iteration)
            # exit()
    print("Training finished")
