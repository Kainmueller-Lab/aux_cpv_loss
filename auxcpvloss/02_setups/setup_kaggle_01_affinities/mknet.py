import sys
import json
import tensorflow as tf

from auxcpvloss import util

def create_network(input_shape, name,
                   output_folder='.',
                   num_fmaps=12,
                   fmap_inc_factors=[4, 4, 4],
                   fmap_dec_factors=[4, 4, 4],
                   downsample_factors=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                   **kwargs):

    tf.reset_default_graph()

    
    if type(input_shape) != tuple:
        input_shape = tuple(input_shape)

    # create a placeholder for the 3D raw input tensor
    raw = tf.placeholder(tf.float32, shape=input_shape, name="raw")

    # create a U-Net
    raw_batched = tf.reshape(raw, (1, 1) + input_shape)
    # unet_output = unet(raw_batched, 14, 4, [[1,3,3],[1,3,3],[1,3,3]])
    model = util.unet(raw_batched, num_fmaps,
                      fmap_inc_factors, fmap_dec_factors,
                      downsample_factors,
                      upsampling="trans_conv", padding="valid")
    print(model)

    # add a convolution layer to create 3 output maps representing affinities
    # in z, y, and x
    pred_batched = util.conv_pass(
        model,
        kernel_size=1,
        num_fmaps=7,
        num_repetitions=2,
        activation=None)

    # get the shape of the output
    output_shape_batched = pred_batched.get_shape().as_list()
    output_shape = output_shape_batched[1:] # strip the batch dimension

    # the 4D output tensor (3, depth, height, width)
    pred = tf.reshape(pred_batched, output_shape)
    pred_affs, pred_bg, pred_cpv = tf.split(pred, [3, 1, 3], 0)

    print(pred_affs, pred_bg, pred_cpv)
    # create a placeholder for the corresponding ground-truth affinities
    gt_affs = tf.placeholder(tf.float32, shape=pred_affs.get_shape(),
                             name="gt_affs")

    gt_cpv = tf.placeholder(tf.float32, shape=pred_cpv.get_shape(),
                            name="gt_cpv")
    gt_labels = tf.placeholder(tf.float32, shape=pred_bg.get_shape(),
                               name="gt_labels")
    gt_bg = tf.clip_by_value(gt_labels, 0, 1)

    # create a placeholder for per-voxel loss weights
    loss_weights_affs = tf.placeholder(
        tf.float32,
        shape=pred_affs.get_shape(),
        name="loss_weights_affs")
    loss_weights_bg = tf.placeholder(
        tf.float32,
        shape=pred_bg.get_shape(),
        name="loss_weights_bg")
    loss_weights_cpv = tf.placeholder(
        tf.float32,
        shape=pred_cpv.get_shape(),
        name="loss_weights_cpv")

    # compute the loss as the weighted mean squared error between the
    # predicted and the ground-truth affinities
    if loss == "mse":
        lossFn = tf.losses.mean_squared_error
        pred_affs = tf.sigmoid(pred_affs)
        pred_bg = tf.sigmoid(pred_bg)
    elif loss == "ce":
        lossFn = tf.losses.sigmoid_cross_entropy

    loss_affs = lossFn(
        gt_affs,
        pred_affs,
        loss_weights_affs)
    if loss == "ce":
        pred_affs = tf.sigmoid(pred_affs)
    if debug:
        loss_affs_debug = lossFn(
            gt_affs,
            pred_affs)
        print_loss_affs_op = tf.print(
            "loss affs:",
            loss_affs,
            loss_affs_debug,
            output_stream=sys.stdout)
        print_loss_weights_affs_op = tf.print(
            "loss weights affs:",
            tf.reduce_sum(loss_weights_affs),
            output_stream=sys.stdout)

    loss_bg = lossFn(
        gt_bg,
        pred_bg,
        loss_weights_bg)
    if loss == "ce":
        pred_bg = tf.sigmoid(pred_bg)
    if debug:
        loss_bg_debug = lossFn(
            gt_bg,
            pred_bg)
        print_loss_bg_op = tf.print("loss bg:", loss_bg, loss_bg_debug,
                                      output_stream=sys.stdout)
        print_loss_weights_bg_op = tf.print("loss weights bg:",
                                              tf.reduce_sum(loss_weights_bg),
                                              output_stream=sys.stdout)

    loss_cpv = tf.losses.mean_squared_error(
        gt_cpv,
        pred_cpv,
        tf.clip_by_value(gt_labels, 0.01, 1.0))
    if debug:
        print_loss_cpv_op = tf.print("loss cpv:", loss_cpv,
                                     output_stream=sys.stdout)

    if debug:
        print_ops = [print_loss_affs_op, print_loss_bg_op, print_loss_cpv_op,
                     print_loss_weights_bg_op, print_loss_weights_affs_op]
    else:
        print_ops = None
    with tf.control_dependencies(print_ops):
        loss = 10.0 * loss_affs + 1.0 * loss_bg# + 25.0 * loss_cpv

    learning_rate = tf.placeholder_with_default(0.1e-3, shape=(),
                                                name="learning-rate")
    # use the Adam optimizer to minimize the loss
    opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    optimizer = opt.minimize(loss)

    # store the network in a meta-graph file
    tf.train.export_meta_graph(filename=name + '.meta')

    # store network configuration for use in train and predict scripts
    config = {
        'raw': raw.name,
        'pred_affs': pred_affs.name,
        'gt_affs': gt_affs.name,
        'gt_labels': gt_labels.name,
        'loss_weights_affs': loss_weights_affs.name,
        'pred_bg': pred_bg.name,
        'gt_bg': gt_bg.name,
        'loss_weights_bg': loss_weights_bg.name,
        'pred_cpv': pred_cpv.name,
        'gt_cpv': gt_cpv.name,
        'loss_weights_cpv': loss_weights_cpv.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'input_shape': input_shape,
        'output_shape': output_shape[1:],
        'output_shape_affs': pred_affs.get_shape().as_list()[1:],
        'output_shape_labels': pred_bg.get_shape().as_list()[1:],
        'output_shape_bg': pred_bg.get_shape().as_list()[1:],
        'output_shape_cpv': pred_cpv.get_shape().as_list()[1:]
    }
    with open(name + '_config.json', 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":
    create_network((230,230,200), 'train_net')
