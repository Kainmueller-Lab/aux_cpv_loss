from __future__ import print_function
import json
import logging
import os
import sys

import gunpowder as gp
import h5py
import numpy as np
import zarr

def predict(**kwargs):
    name = kwargs['name']

    raw = gp.ArrayKey('RAW')
    raw_cropped = gp.ArrayKey('RAW_CROPPED')
    pred_affs = gp.ArrayKey('PRED_AFFS')
    pred_fgbg = gp.ArrayKey('PRED_FGBG')
    pred_cpv = gp.ArrayKey('PRED_CPV')

    with open(os.path.join(kwargs['input_folder'],
                           name + '_config.json'), 'r') as f:
        net_config = json.load(f)
    with open(os.path.join(kwargs['input_folder'],
                           name + '_names.json'), 'r') as f:
        net_names = json.load(f)

    voxel_size = gp.Coordinate(kwargs['voxel_size'])
    input_shape_world = gp.Coordinate(net_config['input_shape'])*voxel_size
    output_shape_world = gp.Coordinate(net_config['output_shape'])*voxel_size
    context = (input_shape_world - output_shape_world)//2

    # formulate the request for what a batch should contain
    request = gp.BatchRequest()
    request.add(raw, input_shape_world)
    request.add(raw_cropped, output_shape_world)
    request.add(pred_affs, output_shape_world)
    request.add(pred_fgbg, output_shape_world)
    request.add(pred_cpv, output_shape_world)

    if kwargs['input_format'] != "hdf" and kwargs['input_format'] != "zarr":
        raise NotImplementedError("predict node for %s not implemented yet",
                                  kwargs['input_format'])
    if kwargs['input_format'] == "hdf":
        sourceNode = gp.Hdf5Source
        with h5py.File(os.path.join(kwargs['data_folder'],
                                    kwargs['sample'] + ".hdf"), 'r') as f:
            shape = f['volumes/raw'].shape
    elif kwargs['input_format'] == "zarr":
        sourceNode = gp.ZarrSource
        f = zarr.open(os.path.join(kwargs['data_folder'],
                                   kwargs['sample'] + ".zarr"), 'r')
        shape = f['volumes/raw'].shape
    source = sourceNode(
        os.path.join(kwargs['data_folder'],
                     kwargs['sample'] + "." + kwargs['input_format']),
        datasets = {
            raw: 'volumes/raw'
        })

    if kwargs['output_format'] != "zarr":
        raise NotImplementedError("Please use zarr as prediction output")
    # pre-create zarr file
    zf = zarr.open(os.path.join(kwargs['output_folder'],
                                kwargs['sample'] + '.zarr'), mode='w')

    # create pred affs dataset
    zf.create('volumes/pred_affs',
              shape=[3] + list(shape),
              chunks=[3] + list(shape),
              dtype=np.float32)
    zf['volumes/pred_affs'].attrs['offset'] = [0, 0, 0]
    zf['volumes/pred_affs'].attrs['resolution'] = kwargs['voxel_size']

    # create pred fgbg dataset
    zf.create('volumes/pred_fgbg',
              shape=[1] + list(shape),
              chunks=[1] + list(shape),
              dtype=np.float32)
    zf['volumes/pred_fgbg'].attrs['offset'] = [0, 0, 0]
    zf['volumes/pred_fgbg'].attrs['resolution'] = kwargs['voxel_size']

    # create pred cpv dataset
    zf.create('volumes/pred_cpv',
              shape=[3] + list(shape),
              chunks=[3] + list(shape),
              dtype=np.float32)
    zf['volumes/pred_cpv'].attrs['offset'] = [0, 0, 0]
    zf['volumes/pred_cpv'].attrs['resolution'] = kwargs['voxel_size']

    # create raw dataset
    zf.create('volumes/raw_cropped',
              shape=[1] + list(shape),
              chunks=[1] + list(shape),
              dtype=np.float32)
    zf['volumes/raw_cropped'].attrs['offset'] = [0, 0, 0]
    zf['volumes/raw_cropped'].attrs['resolution'] = kwargs['voxel_size']


    pipeline = (

        # read from HDF5 file
        source +
        gp.Pad(raw, context) +

        # perform one training iteration for each passing batch (here we use
        # the tensor names earlier stored in train_net.config)
        gp.tensorflow.Predict(
            graph=os.path.join(kwargs['input_folder'], name + '.meta'),
            checkpoint=kwargs['checkpoint'],
            inputs={
                net_names['raw']: raw
            },
            outputs={
                net_names['pred_affs']: pred_affs,
                net_names['pred_fgbg']: pred_fgbg,
                net_names['pred_cpv']: pred_cpv,
                net_names['raw_cropped']: raw_cropped
            }) +

        # store all passing batches in the same HDF5 file
        gp.ZarrWrite(
            {
                raw_cropped: '/volumes/raw_cropped',
                pred_affs: '/volumes/pred_affs',
                pred_fgbg: '/volumes/pred_fgbg',
                pred_cpv: '/volumes/pred_cpv',
            },
            output_dir=kwargs['output_folder'],
            output_filename=kwargs['sample'] + ".zarr",
            compression_type='gzip'
        ) +

        # show a summary of time spend in each node every 10 iterations
        gp.PrintProfilingStats(every=10) +

        # iterate over the whole dataset in a scanning fashion, emitting
        # requests that match the size of the network
        gp.Scan(reference=request)
    )

    with gp.build(pipeline):
        # request an empty batch from Scan to trigger scanning of the dataset
        # without keeping the complete dataset in memory
        pipeline.request_batch(gp.BatchRequest())
