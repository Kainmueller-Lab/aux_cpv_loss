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

    model = util.conv_pass(
        model,
        kernel_size=1,
        num_fmaps=1,
        num_repetitions=1,
        padding=kwargs['padding'],
        activation=None,
        name="output")
    print(model)

    # the 4D output tensor (channels, depth, height, width)
    pred_sdt = tf.squeeze(model, axis=0)
    output_shape = pred_sdt.get_shape().as_list()[1:]

    raw_cropped = util.crop(raw, output_shape)
    raw_cropped = tf.expand_dims(raw_cropped, 0)

    gt_sdt = tf.placeholder(tf.float32, shape=pred_sdt.get_shape(),
                            name="gt_sdt")
    anchor = tf.placeholder(tf.float32, shape=pred_sdt.get_shape(),
                             name="anchor")

    # TODO: check sigmoid, tanh, or nothing? mse/ce?
    loss_sdt, pred_sdt, loss_sdt_print = \
        util.get_loss(gt_sdt, pred_sdt, kwargs['loss'], "sdt", False,
                      do_tanh=kwargs['do_tanh'])

    if kwargs['debug']:
        print_ops = loss_sdt_print
    else:
        print_ops = None
    with tf.control_dependencies(print_ops):
        loss = (1.0 * loss_sdt)

    loss_sum = tf.summary.scalar('loss_sum', loss)
    summaries = tf.summary.merge([loss_sum],
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
        'pred_sdt': pred_sdt.name,
        'gt_sdt': gt_sdt.name,
        'anchor': anchor.name,
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
