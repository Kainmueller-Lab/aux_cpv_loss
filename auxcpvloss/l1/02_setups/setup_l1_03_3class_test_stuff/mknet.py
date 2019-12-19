# coding: utf-8

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

    is_training = tf.placeholder_with_default(False, shape=(),
                                              name="is_training")

    # create a placeholder for the 3D raw input tensor
    raw = tf.placeholder(tf.float32, shape=input_shape, name="raw")

    # create a U-Net
    raw_batched = tf.reshape(raw, (1, 1) + input_shape)
    # unet_output = unet(raw_batched, 14, 4, [[1,3,3],[1,3,3],[1,3,3]])
    model = util.unet_ext(raw_batched,
                          num_fmaps=kwargs['num_fmaps'],
                          fmap_inc_factors=kwargs['fmap_inc_factors'],
                          fmap_dec_factors=kwargs['fmap_dec_factors'],
                          downsample_factors=kwargs['downsample_factors'],
                          activation=kwargs['activation'],
                          padding=kwargs['padding'],
                          kernel_size=kwargs['kernel_size'],
                          num_repetitions=kwargs['num_repetitions'],
                          upsampling=kwargs['upsampling'],
                          batch_norm=kwargs['batch_normalize'],
                          dropout=kwargs['dropout'],
                          is_training=is_training)
    print(model)

    model = util.conv_pass(
        model,
        kernel_size=1,
        num_fmaps=3,
        num_repetitions=1,
        padding=kwargs['padding'],
        activation=None,
        name="output")
    print(model)

    # the 4D output tensor (channels, depth, height, width)
    pred_threeclass = tf.squeeze(model, axis=0)
    output_shape = pred_threeclass.get_shape().as_list()[1:]

    pred_class_max = tf.argmax(pred_threeclass, axis=0, output_type=tf.int32)
    pred_class_max = tf.expand_dims(pred_class_max, 0)
    pred_fgbg = tf.nn.softmax(pred_threeclass, dim=0)[0]
    pred_fgbg = tf.expand_dims(pred_fgbg, 0)

    raw_cropped = util.crop(raw, output_shape)
    raw_cropped = tf.expand_dims(raw_cropped, 0)

    # create a placeholder for the corresponding ground-truth
    gt_threeclass = tf.placeholder(tf.int32, shape=[1]+output_shape,
                               name="gt_threeclass")
    gt_threeclassTmp = tf.squeeze(gt_threeclass, 0)
    anchor = tf.placeholder(tf.float32, shape=gt_threeclass.get_shape(),
                             name="anchor")

    # create a placeholder for per-voxel loss weights
    loss_weights_threeclass = tf.placeholder(
        tf.float32,
        shape=gt_threeclass.get_shape(),
        name="loss_weights_threeclass")

    loss_threeclass, _, loss_threeclass_print = \
        util.get_loss_weighted(gt_threeclassTmp,
                               tf.transpose(pred_threeclass, [1, 2, 3, 0]),
                               tf.transpose(loss_weights_threeclass, [1, 2, 3, 0]),
                               kwargs['loss'], "threeclass", False)

    if kwargs['debug']:
        _, _, loss_threeclass_print2 = \
        util.get_loss(gt_threeclassTmp, tf.transpose(pred_threeclass, [1, 2, 3, 0]),
                      kwargs['loss'], "threeclass", False)
        print_ops = loss_threeclass_print + loss_threeclass_print2
    else:
        print_ops = None
    with tf.control_dependencies(print_ops):
        loss = (1.0 * loss_threeclass)

    loss_sum = tf.summary.scalar('loss_sum', loss)

    if kwargs['cyclic_lr'] or kwargs['warmup_steps']:
        step = tf.get_variable(
            'iteration',
            shape=(),
            initializer=tf.zeros_initializer,
            trainable=False)
        step = tf.assign(
            step,
            step + 1, name="iteration_increment")
        if kwargs['cyclic_lr']:
            step_size = kwargs['cyclic_stepsize']
            max_lr = kwargs['lr'] * 5
            learning_rate = kwargs['lr'] / 10
            gamma = 0.995
            # step = tf.subtract(step, kwargs['cyclic_offset'])
            mode = kwargs['cyclic_mode']
            # computing: cycle = floor( 1 + global_step / (2*step_size ) )
            double_step = tf.multiply(2., step_size)
            global_div_double_step = tf.divide(step, double_step)
            cycle = tf.floor(tf.add(1., global_div_double_step))
            # computing: x = abs( global_step / step_size – 2 * cycle +1 )
            double_cycle = tf.multiply(2., cycle)
            global_div_step = tf.divide(step, step_size)
            tmp = tf.subtract(global_div_step, double_cycle)
            x = tf.abs(tf.add(1., tmp))
            # computing: clr = learning_rate + ( max_lr – learning_rate )
            #                   * max( 0, 1 - x )
            a1 = tf.maximum(0., tf.subtract(1., x))
            a2 = tf.subtract(max_lr, learning_rate)
            clr = tf.multiply(a1, a2)
            if mode == 'triangular2':
                clr = tf.divide(clr, tf.cast(tf.pow(2, tf.cast(
                    cycle-1, tf.int32)), tf.float32))
            if mode == 'exp_range':
                clr = tf.multiply(tf.pow(gamma, step//100), clr)
            cyclic_lr = tf.add(clr, learning_rate, name="cyclic_lr")
            cyclic_lr = tf.Print(cyclic_lr, [cyclic_lr], "cyclic_lr", first_n=10)
            print(cyclic_lr)
        if kwargs['warmup_steps']:
            arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
            arg2 = step * (tf.cast(kwargs['warmup_steps'],tf.float32) ** -1.5)
            fac = tf.math.rsqrt(tf.cast(kwargs['warmup_fac'], tf.float32))

            warmup_lr = tf.multiply(fac, tf.math.minimum(arg1, arg2),
                                    name="warmup_lr")
            warmup_lr = tf.Print(warmup_lr, [warmup_lr], "warmup_lr", first_n=10)
            print(warmup_lr)
        if kwargs['cyclic_lr'] and kwargs['warmup_steps']:
            learning_rate = tf.cond(tf.less(step, kwargs['warmup_steps']),
                                    lambda: warmup_lr,
                                    lambda: cyclic_lr,
                                    name="learning-rate")
            print(learning_rate)
        elif kwargs['cyclic_lr']:
            learning_rate = tf.identity(cyclic_lr, name="learning-rate")
        elif kwargs['warmup_steps']:
            learning_rate = tf.identity(warmup_lr, name="learning-rate")
    else:
        learning_rate = tf.placeholder_with_default(kwargs['lr'], shape=(),
                                                    name="learning-rate")
    print(learning_rate)
    lr_sum = tf.summary.scalar('lr_sum', learning_rate)

    # use the Adam optimizer to minimize the loss
    if kwargs['optimizer'] == 'Adam':
        opt = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.95,
            beta2=0.999,
            epsilon=1e-8)
    elif kwargs['optimizer'] == 'SGD':
        opt = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)
    elif kwargs['optimizer'] == 'momentum':
        opt = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=kwargs['momentum'],
            use_nesterov=kwargs['use_nesterov'])
    else:
        raise RuntimeError("invalid value for optimizer {}".format(
            kwargs['optimizer']))
    print(opt)
    optimizer = opt.minimize(loss)

    summaries = tf.summary.merge([loss_sum, lr_sum], name="summaries")

    # store the network in a meta-graph file
    tf.train.export_meta_graph(filename=os.path.join(kwargs['output_folder'],
                                                     kwargs['name'] +'.meta'))

    # store network configuration for use in train and predict scripts
    fn = os.path.join(kwargs['output_folder'], kwargs['name'])
    names = {
        'raw': raw.name,
        'raw_cropped': raw_cropped.name,
        'gt_threeclass': gt_threeclass.name,
        'pred_threeclass': pred_threeclass.name,
        'pred_class_max': pred_class_max.name,
        'pred_fgbg': pred_fgbg.name,
        'anchor': anchor.name,
        'loss_weights_threeclass': loss_weights_threeclass.name,
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
