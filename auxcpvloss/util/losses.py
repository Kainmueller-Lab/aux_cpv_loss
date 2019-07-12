import sys

import tensorflow as tf


def get_loss_fn(loss):
    if loss == "mse":
        loss_fn = tf.losses.mean_squared_error
    elif loss == "ce":
        loss_fn = tf.losses.sigmoid_cross_entropy
    else:
        raise ValueError("invalid loss function", loss)
    return loss_fn


def get_aff_loss(gt_affs, pred_affs, loss_weights_affs, loss):
    loss_fn = get_loss_fn(loss)

    if loss == "mse":
        pred_affs = tf.sigmoid(pred_affs)
    loss_affs_weighted = loss_fn(
        gt_affs,
        pred_affs,
        loss_weights_affs)
    if loss == "ce":
        pred_affs = tf.sigmoid(pred_affs)

    loss_affs_nonweighted = loss_fn(
        gt_affs,
        pred_affs)
    print_loss_affs_op = tf.print(
        "affinities-loss (weighted/nonweighted):",
        loss_affs_weighted,
        loss_affs_nonweighted,
        output_stream=sys.stdout)
    print_loss_affs_weights_op = tf.print(
        "affinities-loss weights:",
        tf.reduce_sum(loss_weights_affs),
        output_stream=sys.stdout)
    return loss_affs_weighted, pred_affs, \
        [print_loss_affs_op, print_loss_affs_weights_op]


def get_fgbg_loss(gt_fgbg, pred_fgbg, loss_weights_fgbg, loss):
    loss_fn = get_loss_fn(loss)

    if loss == "mse":
        pred_fgbg = tf.sigmoid(pred_fgbg)
    loss_fgbg_weighted = loss_fn(
        gt_fgbg,
        pred_fgbg,
        loss_weights_fgbg)
    if loss == "ce":
        pred_fgbg = tf.sigmoid(pred_fgbg)

    loss_fgbg_nonweighted = loss_fn(
        gt_fgbg,
        pred_fgbg)
    print_loss_fgbg_op = tf.print(
        "fgbg-loss (weighted/nonweighted):",
        loss_fgbg_weighted,
        loss_fgbg_nonweighted,
        output_stream=sys.stdout)
    print_loss_fgbg_weights_op = tf.print(
        "fgbg-loss weights:",
        tf.reduce_sum(loss_weights_fgbg),
        output_stream=sys.stdout)
    return loss_fgbg_weighted, pred_fgbg, \
        [print_loss_fgbg_op, print_loss_fgbg_weights_op]


def get_cpv_loss(gt_cpv, pred_cpv, loss_weights_cpv):
    loss_fn = get_loss_fn("mse")

    loss_cpv_weighted = loss_fn(
        gt_cpv,
        pred_cpv,
        loss_weights_cpv)

    loss_cpv_nonweighted = loss_fn(
        gt_cpv,
        pred_cpv)
    print_loss_cpv_op = tf.print(
        "cpv-loss (weighted/nonweighted):",
        loss_cpv_weighted,
        loss_cpv_nonweighted,
        output_stream=sys.stdout)
    print_loss_cpv_weights_op = tf.print(
        "cpv-loss weights:",
        tf.reduce_sum(loss_weights_cpv),
        output_stream=sys.stdout)
    return loss_cpv_weighted, \
        [print_loss_cpv_op, print_loss_cpv_weights_op]
