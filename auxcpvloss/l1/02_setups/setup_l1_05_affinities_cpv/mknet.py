import sys
import os
import json
import tensorflow as tf

from auxcpvloss import util


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
    model = util.unet(raw_batched,
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
    model = util.conv_pass(
        model,
        kernel_size=1,
        num_fmaps=7,
        num_repetitions=1,
        padding=kwargs['padding'],
        activation=None,
        name="output")
    print(model)

    # the 4D output tensor (channels, depth, height, width)
    pred = tf.squeeze(model, axis=0)
    output_shape = pred.get_shape().as_list()[1:]
    pred_affs, pred_fgbg, pred_cpv = tf.split(pred, [3, 1, 3], 0)

    raw_cropped = util.crop(raw, output_shape)
    raw_cropped = tf.expand_dims(raw_cropped, 0)

    # create a placeholder for the corresponding ground-truth affinities
    gt_affs = tf.placeholder(tf.float32, shape=pred_affs.get_shape(),
                             name="gt_affs")
    gt_cpv = tf.placeholder(tf.float32, shape=pred_cpv.get_shape(),
                            name="gt_cpv")
    gt_labels = tf.placeholder(tf.float32, shape=pred_fgbg.get_shape(),
                               name="gt_labels")
    gt_fgbg = tf.placeholder(tf.float32, shape=pred_fgbg.get_shape(),
                             name="gt_fgbg")
    anchor = tf.placeholder(tf.float32, shape=pred_fgbg.get_shape(),
                             name="anchor")
    # gt_fgbg = tf.clip_by_value(gt_labels, 0, 1)

    # create a placeholder for per-voxel loss weights
    loss_weights_affs = tf.placeholder(
        tf.float32,
        shape=pred_affs.get_shape(),
        name="loss_weights_affs")
    loss_weights_fgbg = tf.placeholder(
        tf.float32,
        shape=pred_fgbg.get_shape(),
        name="loss_weights_fgbg")
    loss_weights_cpv = tf.placeholder(
        tf.float32,
        shape=pred_cpv.get_shape(),
        name="loss_weights_cpv")

    # compute the loss as the weighted mean squared error between the
    # predicted and the ground-truth affinities
    loss_affs, pred_affs, loss_affs_print = \
        util.get_loss_weighted(gt_affs, pred_affs, loss_weights_affs,
                               kwargs['loss'], "affinities", True)
    loss_fgbg, pred_fgbg, loss_fgbg_print = \
        util.get_loss_weighted(gt_fgbg, pred_fgbg, loss_weights_fgbg,
                               kwargs['loss'], "fgbg", True)
    loss_cpv, pred_cpv, loss_cpv_print = \
        util.get_loss_weighted(gt_cpv, pred_cpv,
                               tf.clip_by_value(gt_labels, 0.01, 1.0),
                               'mse', 'cpv', False)

    if kwargs['debug']:
        _, _, loss_affs_print2 = \
        util.get_loss(gt_affs, pred_affs, kwargs['loss'], "affinities", True)
        _, _, loss_fgbg_print2 = \
        util.get_loss(gt_fgbg, pred_fgbg, kwargs['loss'], "fgbg", True)
        _, _, loss_cpv_print2 = \
        util.get_loss(gt_cpv, pred_cpv, 'mse', 'cpv', False)

        print_ops = loss_affs_print + loss_affs_print2 + \
                    loss_fgbg_print + loss_fgbg_print2 + \
                    loss_cpv_print + loss_cpv_print2
    else:
        print_ops = None
    with tf.control_dependencies(print_ops):
        loss = (kwargs['loss_affs_coeff'] * loss_affs +
                kwargs['loss_fgbg_coeff'] * loss_fgbg +
                kwargs['loss_cpv_coeff']  * loss_cpv)

    loss_aff_sum = tf.summary.scalar('loss_aff_sum',
                                     kwargs['loss_affs_coeff'] * loss_affs)
    loss_fgbg_sum = tf.summary.scalar('loss_fgbg_sum',
                                     kwargs['loss_fgbg_coeff'] * loss_fgbg)
    loss_cpv_sum = tf.summary.scalar('loss_cpv_sum',
                                     kwargs['loss_cpv_coeff'] * loss_cpv)
    loss_sum = tf.summary.scalar('loss_sum', loss)
    summaries = tf.summary.merge([loss_aff_sum, loss_fgbg_sum,
                                  loss_cpv_sum, loss_sum], name="summaries")

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
        'loss_weights_affs': loss_weights_affs.name,
        'pred_fgbg': pred_fgbg.name,
        'gt_fgbg': gt_fgbg.name,
        'anchor': anchor.name,
        'loss_weights_fgbg': loss_weights_fgbg.name,
        'pred_cpv': pred_cpv.name,
        'gt_cpv': gt_cpv.name,
        'loss_weights_cpv': loss_weights_cpv.name,
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
