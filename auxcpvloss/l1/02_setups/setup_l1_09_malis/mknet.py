import sys
import os
import json
import tensorflow as tf

import malis

from auxcpvloss import util
from funlib.learn.tensorflow.models import unet, conv_pass, crop

def mk_net(**kwargs):

    tf.reset_default_graph()

    input_shape = kwargs['input_shape']
    if not isinstance(input_shape, tuple):
        input_shape = tuple(input_shape)

    # create a placeholder for the 3D raw input tensor
    raw = tf.placeholder(tf.float32, shape=input_shape, name="raw")

    # create a U-Net
    raw_batched = tf.reshape(raw, (1, 1) + input_shape)
    # unet_output = unet(raw_batched, 14, 4, [[1,3,3],[1,3,3],[1,3,3]])
    model, _, _ = unet(raw_batched,
                       num_fmaps=kwargs['num_fmaps'],
                       fmap_inc_factors=kwargs['fmap_inc_factors'],
                       fmap_dec_factors=kwargs['fmap_dec_factors'],
                       downsample_factors=kwargs['downsample_factors'],
                       activation=kwargs['activation'],
                       padding=kwargs['padding'],
                       kernel_size=kwargs['kernel_size'],
                       num_repetitions=kwargs['num_repetitions'],
                       upsampling=kwargs['upsampling'])
    print(model)

    # add a convolution layer to create 3 output maps representing affinities
    # in z, y, and x
    model, _ = conv_pass(
        model,
        kernel_sizes=[1],
        num_fmaps=4,
        # num_repetitions=1,
        padding=kwargs['padding'],
        activation=None,
        name="output")
    print(model)

    # the 4D output tensor (channels, depth, height, width)
    pred = tf.squeeze(model, axis=0)
    output_shape = pred.get_shape().as_list()[1:]
    pred_affs, pred_fgbg = tf.split(pred, [3, 1], 0)

    raw_cropped = crop(raw, output_shape)
    raw_cropped = tf.expand_dims(raw_cropped, 0)

    # create a placeholder for the corresponding ground-truth affinities
    gt_affs = tf.placeholder(tf.float32, shape=pred_affs.get_shape(),
                             name="gt_affs")
    gt_labels = tf.placeholder(tf.float32, shape=pred_fgbg.get_shape(),
                               name="gt_labels")
    gt_fgbg = tf.placeholder(tf.float32, shape=pred_fgbg.get_shape(),
                             name="gt_fgbg")
    anchor = tf.placeholder(tf.float32, shape=pred_fgbg.get_shape(),
                             name="anchor")
    # gt_fgbg = tf.clip_by_value(gt_labels, 0, 1)

    # create a placeholder for per-voxel loss weights
    # loss_weights_affs = tf.placeholder(
    #     tf.float32,
    #     shape=pred_affs.get_shape(),
    #     name="loss_weights_affs")
    loss_weights_fgbg = tf.placeholder(
        tf.float32,
        shape=pred_fgbg.get_shape(),
        name="loss_weights_fgbg")

    # compute the loss as the weighted mean squared error between the
    # predicted and the ground-truth affinities
    loss_fgbg, pred_fgbg, loss_fgbg_print = \
        util.get_loss_weighted(gt_fgbg, pred_fgbg, loss_weights_fgbg,
                               kwargs['loss'], "fgbg", True)

    neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    gt_seg = tf.squeeze(gt_labels, axis=0, name="gt_seg")
    pred_affs = tf.identity(pred_affs, name="pred_affs")
    loss_malis = malis.malis_loss_op(pred_affs, gt_affs, gt_seg,
                                     neighborhood, name="malis_loss")

    loss = (kwargs['loss_malis_coeff'] * loss_malis +
            kwargs['loss_fgbg_coeff'] * loss_fgbg)


    loss_malis_sum = tf.summary.scalar('loss_malis_sum',
                                       kwargs['loss_malis_coeff'] * loss_malis)
    loss_fgbg_sum = tf.summary.scalar('loss_fgbg_sum',
                                     kwargs['loss_fgbg_coeff'] * loss_fgbg)
    loss_sum = tf.summary.scalar('loss_sum', loss)
    summaries = tf.summary.merge([loss_malis_sum, loss_fgbg_sum, loss_sum],
                                 name="summaries")

    learning_rate = tf.placeholder_with_default(kwargs['lr'], shape=(),
                                                name="learning-rate")
    # use the Adam optimizer to minimize the loss
    opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    optimizer = opt.minimize(loss)

    # store the network in a meta-graph file
    tf.train.export_meta_graph(filename=os.path.join(kwargs['output_folder'],
                                                     kwargs['name'] +'.meta'))

    # store network configuration for use in train and predict scripts
    fn = os.path.join(kwargs['output_folder'], kwargs['name'])
    names = {
        'raw': raw.name,
        'raw_cropped': raw_cropped.name,
        'pred_affs': pred_affs.name,
        'gt_affs': gt_affs.name,
        'gt_labels': gt_labels.name,
        # 'loss_weights_affs': loss_weights_affs.name,
        'pred_fgbg': pred_fgbg.name,
        'gt_fgbg': gt_fgbg.name,
        'anchor': anchor.name,
        'loss_weights_fgbg': loss_weights_fgbg.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'summaries': summaries.name
    }

    with open(fn + '_names.json', 'w') as f:
        json.dump(names, f)

    config = {
        'input_shape': input_shape,
        'output_shape': output_shape,
    }
    with open(fn + '_config.json', 'w') as f:
        json.dump(config, f)
