import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST_ADD'] = 'Conv3D,Conv3DBackpropFilter,Conv3DBackpropFilterV2,Conv3DBackpropInput,Conv3DBackpropInputV2,Conv2D,Conv2DBackpropFilter,Conv2DBackpropFilterV2,Conv2DBackpropInput,Conv2DBackpropInputV2'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_CPP_VMODULE'] = 'amp_optimizer=2'
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
import argparse
from datetime import datetime
from glob import glob
import fnmatch
import functools
import importlib
import itertools
import json
import logging

try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)
from multiprocessing import Process
import operator
import os
import random
import runpy
import shutil
import sys
import time

import h5py
from joblib import Parallel, delayed
from natsort import natsorted
import numpy as np
import pandas as pd
import toml
import zarr

import util
import evaluateInstanceSegmentation as eval_seg
import evaluateInstanceDetection as eval_det


def merge_dicts(sink, source):
    if not isinstance(sink, dict) or not isinstance(source, dict):
        raise TypeError('Args to merge_dicts should be dicts')

    for k, v in source.items():
        if isinstance(source[k], dict) and isinstance(sink.get(k), dict):
            sink[k] = merge_dicts(sink[k], v)
        else:
            sink[k] = v

    return sink


def backup_and_copy_file(source, target, fn):
    target = os.path.join(target, fn)
    if os.path.exists(target):
        os.replace(target, target + "_backup" + str(int(time.time())))
    if source is not None:
        source = os.path.join(source, fn)
        shutil.copy2(source, target)

def check_file(fn, remove_on_error=False, key=None):
    if fn.endswith("zarr"):
        try:
            fl = zarr.open(fn, 'r')
            if key is not None:
                tmp = fl[key]
            return True
        except Exception as e:
            logger.info("%s", e)
            if remove_on_error:
                shutil.rmtree(fn, ignore_errors=True)
            return False
    elif fn.endswith("hdf"):
        try:
            with h5py.File(fn, 'r') as fl:
                if key is not None:
                    tmp = fl[key]
            return True
        except Exception as e:
            if remove_on_error:
                os.remove(fn)
            return False
    else:
        raise NotImplementedError("invalid file type")

def time_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = datetime.now()
        ret = func(*args, **kwargs)
        logger.info('time %s: %s', func.__name__, str(datetime.now() - t0))
        return ret
    return wrapper


def fork(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            p = Process(target=func, args=args, kwargs=kwargs)
            p.start()
            p.join()
            if p.exitcode != 0:
                raise RuntimeError("child process died")
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            p.terminate()
            p.join()
            os._exit(-1)

    return wrapper

logger = logging.getLogger(__name__)


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', action='append',
                        help=('Configuration files to use. For defaults, '
                              'see `config/default.toml`.'))
    parser.add_argument('-a', '--app', dest='app',
                        help=('Application to use. Choose out of cityscapes, '
                              'flylight, kaggle, etc.'))
    parser.add_argument('-r', '--root', dest='root', default=None,
                        help='Experiment folder to store results.')
    parser.add_argument('-s', '--setup', dest='setup', default=None,
                        help='Setup for experiment.', required=True)
    parser.add_argument('-id', '--exp-id', dest='expid', default=None,
                        help='ID for experiment.')

    # action options
    parser.add_argument('-d', '--do', dest='do', default=[], nargs='+',
                        choices=['all',
                                 'mknet',
                                 'train',
                                 'predict',
                                 'label',
                                 'validate_checkpoints',
                                 'validate',
                                 'postprocess',
                                 'evaluate',
                                 'visualize'
                                 ],
                        help='Task to do for experiment.')

    parser.add_argument('--test-checkpoint', dest='test_checkpoint',
                        default='last', choices=['last', 'best'],
                        help=('Specify which checkpoint to use for testing. '
                              'Either last or best (checkpoint validation).'))

    parser.add_argument('--checkpoint', dest='checkpoint', default=None,
                        type=int,
                        help='Specify which checkpoint to use.')

    parser.add_argument("--run_from_exp", action="store_true",
                        help='run from setup or from experiment folder')
    parser.add_argument("--validate_on_train", action="store_true",
                        help=('validate using training data'
                              '(to check for overfitting)'))
    parser.add_argument("--do_detection", action="store_true",
                        help='detection instead of segmentation for evaluation')
    parser.add_argument("--param_product", action="store_true",
                        help='build product of validation params (instead of zipped list)')

    # train / val / test datasets
    parser.add_argument('--input-format', dest='input_format',
                        choices=['hdf', 'zarr', 'n5', 'tif'],
                        help='File format of dataset.')
    parser.add_argument('--train-data', dest='train_data', default=None,
                        help='Train dataset to use.')
    parser.add_argument('--val-data', dest='val_data', default=None,
                        help='Validation dataset to use.')
    parser.add_argument('--test-data', dest='test_data', default=None,
                        help='Test dataset to use.')

    parser.add_argument("--debug_args", action="store_true",
                        help=('Set some low values to certain'
                              ' args for debugging.'))
    parser.add_argument("--term_after_predict", action="store_true",
                        help=('terminate after prediction'
                              '(to split job into gpu/non-gpu part)'))

    parser.add_argument('-m', '--trained_model', action='append',
                        help=('Models to average and find best params'
                              'train on whole train set using those params'))

    args = parser.parse_args()

    return args


def create_folders(args, filebase):
    if (args.expid is None or \
        not os.path.isdir(os.path.join(filebase, 'train'))) and \
        args.run_from_exp:
        setup = os.path.join(args.app, '02_setups', args.setup)
        os.makedirs(filebase, exist_ok=True)
        backup_and_copy_file(setup, filebase, 'train.py')
        backup_and_copy_file(setup, filebase, 'mknet.py')
        backup_and_copy_file(setup, filebase, 'predict.py')
        backup_and_copy_file(setup, filebase, 'label.py')



    # create train folders
    train_folder = os.path.join(filebase, 'train')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'snapshots'), exist_ok=True)
    backup_and_copy_file(setup, train_folder, 'train_net_config.json')
    backup_and_copy_file(setup, train_folder, 'train_net_names.json')

    # create val folders
    val_folder = os.path.join(filebase, 'val')
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(os.path.join(val_folder, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(val_folder, 'instanced'), exist_ok=True)

    # create test folders
    test_folder = os.path.join(filebase, 'test')
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'instanced'), exist_ok=True)

    return train_folder, val_folder, test_folder


def update_config(args, config):
    if args.train_data is not None:
        config['data']['train_data'] = args.train_data

    if args.val_data is not None:
        config['data']['val_data'] = args.val_data

    if args.test_data is not None:
        config['data']['test_data'] = args.test_data

    if args.input_format is not None:
        config['data']['input_format'] = args.input_format
    if 'input_format' not in config['data']:
        raise ValueError("Please provide data/input_format in cl or config")

    if args.validate_on_train:
        config['data']['validate_on_train'] = True
    else:
        config['data']['validate_on_train'] = False


def setDebugValuesForConfig(config):
    config['training']['max_iterations'] = 10
    config['training']['checkpoints'] = 10
    config['training']['snapshots'] = 10
    config['training']['profiling'] = 10
    config['training']['num_workers'] = 1
    config['training']['cache_size'] = 1


@fork
@time_func
def mknet(args, config, train_folder, test_folder):
    if args.run_from_exp:
        mk_net_fn = runpy.run_path(
            os.path.join(config['base'], 'mknet.py'))['mk_net']
    else:
        mk_net_fn = importlib.import_module(
            args.app + '.02_setups.' + args.setup + '.mknet').mk_net

    mk_net_fn(name=config['model']['train_net_name'],
              input_shape=config['model']['train_input_shape'],
              output_folder=train_folder,
              **config['model'],
              **config['optimizer'],
              debug=config['general']['debug'])
    mk_net_fn(name=config['model']['test_net_name'],
              input_shape=config['model']['test_input_shape'],
              output_folder=test_folder,
              **config['model'],
              **config['optimizer'],
              debug=config['general']['debug'])


@fork
@time_func
def train(args, config, train_folder):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError("no free GPU available!")

    data_files = get_list_train_files(config)
    if args.run_from_exp:
        train_fn = runpy.run_path(
            os.path.join(config['base'], 'train.py'))['train_until']
    else:
        train_fn = importlib.import_module(
            args.app + '.02_setups.' + args.setup + '.train').train_until

    train_fn(name=config['model']['train_net_name'],
             max_iteration=config['training']['max_iterations'],
             output_folder=train_folder,
             data_files=data_files,
             **config['data'],
             **config['training'],
             **config['optimizer'],
             **config['preprocessing'])


def get_list_train_files(config):
    data = config['data']['train_data']
    if os.path.isfile(data):
        files = [data]
    elif os.path.isdir(data):
        if 'folds' in config['training']:
            files = glob(os.path.join(
                data + "_folds" + config['training']['folds'],
                "*." + config['data']['input_format']))
        else:
            files = glob(os.path.join(data,
                                      "*." + config['data']['input_format']))
    else:
        raise ValueError(
            "please provide file or directory for data/train_data", data)
    return files


def get_list_samples(config, data, file_format):
    logger.info("reading data from %s", data)
    # read data
    if os.path.isfile(data):
        if file_format == ".hdf" or file_format == "hdf":
            with h5py.File(data, 'r') as f:
                samples = [k for k in f]
        else:
            NotImplementedError("Add reader for %s format",
                                os.path.splitext(data)[1])
    elif os.path.isdir(data):
        samples = fnmatch.filter(os.listdir(data),
                                 '*.' + file_format)
        samples = [os.path.splitext(s)[0] for s in samples]
    else:
        raise NotImplementedError("Data must be file or directory")

    return samples


@fork
def predict_sample(args, config, name, data, sample, checkpoint, input_folder,
                   output_folder):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError("no free GPU available!")

    if args.run_from_exp:
        predict_fn = runpy.run_path(
            os.path.join(config['base'], 'predict.py'))['predict']
    else:
        predict_fn = importlib.import_module(
            args.app + '.02_setups.' + args.setup + '.predict').predict

    logger.info('predicting %s!', sample)
    predict_fn(name=name, sample=sample, checkpoint=checkpoint,
               data_folder=data, input_folder=input_folder,
               output_folder=output_folder,
               **config['data'],
               **config['model'],
               **config['preprocessing'],
               **config['prediction'])


@time_func
def predict(args, config, name, data, checkpoint, test_folder, output_folder):

    samples = get_list_samples(config, data, config['data']['input_format'])

    for idx, sample in enumerate(samples):
        fl = os.path.join(output_folder,
                          sample + '.' + config['prediction']['output_format'])
        if not config['general']['overwrite'] and os.path.exists(fl):
            if check_file(fl, remove_on_error=True, key=config['postprocessing']['surf_key']):
                logger.info('Skipping prediction for %s (%s). Already exists!',
                            sample, config['postprocessing']['surf_key'])
                continue
            else:
                logger.info('prediction %s (%s) broken. recomputing..!',
                            sample, config['postprocessing']['surf_key'])

        if args.debug_args and idx >= 2:
            break

        predict_sample(args, config, name, data, sample, checkpoint,
                       test_folder, output_folder)


def get_checkpoint_file(iteration, name, train_folder):
    return os.path.join(train_folder, name + '_checkpoint_%d' % iteration)


def get_checkpoint_list(name, train_folder):
    checkpoints = natsorted(glob(
        os.path.join(train_folder, name + '_checkpoint_*.index')))
    return [int(os.path.splitext(os.path.basename(cp))[0].split("_")[-1])
            for cp in checkpoints]


def select_validation_data(config, train_folder, val_folder):
    if config['data'].get('validate_on_train'):
        if 'folds' in config['training']:
            data = config['data']['train_data'] + \
                   "_folds" + str(config['training']['folds'])
        else:
            data = config['data']['train_data']
        output_folder = train_folder
    else:
        if 'fold' in config['validation']:
            data = config['data']['val_data'] + \
                   "_fold" + str(config['validation']['fold'])
        else:
            data = config['data']['val_data']
        output_folder = val_folder
    return data, output_folder


@time_func
def validate_checkpoint(args, config, data, checkpoint, params, train_folder,
                        test_folder, output_folder):
    logger.info("validating checkpoint %d %s", checkpoint, params)

    # create test iteration folders
    pred_folder = os.path.join(output_folder, 'processed', str(checkpoint))
    os.makedirs(pred_folder, exist_ok=True)

    # predict val data
    checkpoint_file = get_checkpoint_file(checkpoint,
                                          config['model']['train_net_name'],
                                          train_folder)
    logger.info("predicting checkpoint %d", checkpoint)
    predict(args, config, config['model']['test_net_name'], data,
            checkpoint_file, test_folder, pred_folder)

    if not params:
        return

    params_str = [k + "_" + str(v).replace(".", "_")
                  for k, v in params.items()]
    inst_folder = os.path.join(output_folder, 'instanced', str(checkpoint),
                               *params_str)
    detect_str = "detection" \
                     if config['evaluation'].get("do_detection", False) or \
                        args.do_detection \
                     else ""
    eval_folder = os.path.join(output_folder, 'evaluated',
                               detect_str,
                               str(checkpoint), *params_str)
    os.makedirs(inst_folder, exist_ok=True)
    os.makedirs(eval_folder, exist_ok=True)

    # label
    logger.info("labelling checkpoint %d %s", checkpoint, params)
    label(args, config, data, pred_folder, inst_folder, params)

    # evaluate
    logger.info("evaluating checkpoint %d %s", checkpoint, params)
    metric = evaluate(args, config, data, inst_folder, eval_folder)
    logger.info("%s checkpoint %6d: %.4f (%s)",
                config['evaluation']['metric'], checkpoint, metric, params)

    return metric


def get_postprocessing_params(config, params_list, test_config):
    params = {}
    for p in params_list:
        if config is None or config[p] == []:
            params[p] = [test_config[p]]
        else:
            params[p] = config[p]

    return params


def named_product(**items):
    if items:
        names = items.keys()
        vals = items.values()
        for res in itertools.product(*vals):
            yield dict(zip(names, res))
    else:
        yield {}

def named_zip(**items):
    if items:
        names = items.keys()
        vals = items.values()
        for res in zip(*vals):
            yield dict(zip(names, res))
    else:
        yield {}


def validate_checkpoints(args, config, data, checkpoints, train_folder,
                         test_folder, output_folder):
    # validate all checkpoints and return best one
    metrics = []
    ckpts = []
    params = []
    results = []
    if args.param_product:
        get_param_func = named_product
    else:
        get_param_func = named_zip
    param_sets = list(get_param_func(
        **get_postprocessing_params(
            config['validation'],
            config['validation'].get('params', []),
            config['postprocessing']
        )))

    # only predict (params=None)
    for checkpoint in checkpoints:
        validate_checkpoint(args, config, data, checkpoint, None,
                            train_folder, test_folder, output_folder)

    if args.term_after_predict:
        exit(0)

    # label and eval
    for checkpoint in checkpoints:
        num_workers = config['validation'].get("num_workers", 1)
        res = Parallel(n_jobs=num_workers, backend='multiprocessing') \
            (delayed(validate_checkpoint)(
                args, config, data, checkpoint, p, train_folder, test_folder,
                output_folder)
             for p in param_sets)
        for idx, metric in enumerate(res):
            metrics.append(metric)
            ckpts.append(checkpoint)
            params.append(param_sets[idx])
            results.append({'checkpoint': checkpoint,
                            'metric': metric,
                            'params': param_sets[idx]})

    for ch, metric, p in zip(ckpts, metrics, params):
        logger.info("%s checkpoint %6d: %.4f (%s)",
                    config['evaluation']['metric'], ch, metric, p)

    if config['general']['debug'] and None in metrics:
        logger.error("None in checkpoint found: %s (continuing with last)",
                     tuple(metrics))
        best_checkpoint = ckpts[-1]
        best_params = params[-1]
    else:
        best_checkpoint = ckpts[np.nanargmax(metrics)]
        best_params = params[np.nanargmax(metrics)]
    logger.info('best checkpoint: %d', best_checkpoint)
    logger.info('best params: %s', best_params)
    with open(os.path.join(output_folder, "results.json"), 'w') as f:
        json.dump(results, f)
    return best_checkpoint, best_params


def label_sample(args, config, data, pred_folder, output_folder, params,
                 sample):
    if args.run_from_exp:
        label = runpy.run_path(
            os.path.join(config['base'], 'label.py'))['label']
    else:
        label = importlib.import_module(
            args.app + '.02_setups.' + args.setup + '.label').label

    output_fn = os.path.join(
        output_folder, sample + '.' + \
        config['postprocessing']['output_format'])
    if not config['general']['overwrite'] and \
       os.path.exists(output_fn):
        if check_file(output_fn, remove_on_error=True,
                      key=config['evaluation']['res_key']):
            logger.info('Skipping labelling for %s (%s). Already exists!',
                        sample, config['evaluation']['res_key'])
            return
        else:
            logger.info('labelling %s (%s) broken. recomputing..!',
                        sample, config['evaluation']['res_key'])

    if config['postprocessing']['include_gt']:
        gt = os.path.join(data, sample + "." + config['data']['input_format'])
    else:
        gt = None

    config['postprocessing'] = merge_dicts(config['postprocessing'], params)
    label(sample=sample, gt=gt, pred_folder=pred_folder,
          output_folder=output_folder,
          output_fn=output_fn,
          pred_format=config['prediction']['output_format'],
          gt_format=config['data']['input_format'],
          gt_key=config['data']['gt_key'],
          debug=config['general']['debug'],
          **config['postprocessing'])


@time_func
def label(args, config, data, pred_folder, output_folder, params):
    samples = get_list_samples(config, pred_folder,
                               config['prediction']['output_format'])
    num_workers = config['postprocessing'].get("num_workers", 1)
    if num_workers > 1:
        Parallel(n_jobs=num_workers, backend='multiprocessing', verbose=1) \
            (delayed(label_sample)(args, config, data, pred_folder,
                                   output_folder, params, s) for s in samples)
    else:
        for sample in samples:
            label_sample(args, config, data, pred_folder,
                         output_folder, params, sample)


def evaluate_sample(args, config, data, sample, inst_folder, output_folder,
                    file_format):
    if os.path.isfile(data):
        gt_path = data
        gt_key = sample + "/gt"
    else:
        gt_path = os.path.join(
            data, sample + "." + config['data']['input_format'])
        gt_key = config['data']['gt_key']

    sample_path = os.path.join(inst_folder, sample + "." + file_format)

    if config['evaluation'].get("do_detection", False) or args.do_detection:
        eval_fn = eval_det.evaluate_file
    else:
        eval_fn = eval_seg.evaluate_file
    return eval_fn(res_file=sample_path, gt_file=gt_path,
                   gt_key=gt_key,
                   out_dir=output_folder, suffix="",
                   **config['evaluation'],
                   debug=config['general']['debug'])


@time_func
def evaluate(args, config, data, inst_folder, output_folder, return_avg=True,
             returnIoUavg=None):
    file_format = config['postprocessing']['output_format']
    samples = natsorted(get_list_samples(config, inst_folder, file_format))

    num_workers = config['evaluation'].get("num_workers", 1)
    if num_workers > 1:
        metric_dicts = Parallel(n_jobs=num_workers, backend='multiprocessing',
                                verbose=0)(
            delayed(evaluate_sample)(args, config, data, s, inst_folder,
                                     output_folder, file_format)
            for s in samples)
    else:
        metric_dicts = []
        for sample in samples:
            metric_dict = evaluate_sample(args, config, data, sample, inst_folder,
                                          output_folder, file_format)
            metric_dicts.append(metric_dict)

    metrics = {}
    metrics_full = {}
    if returnIoUavg is None:
        for metric_dict, sample in zip(metric_dicts, samples):
            if metric_dict is None:
                continue
            metrics_full[sample] = metric_dict
            for k in config['evaluation']['metric'].split('.'):
                metric_dict = metric_dict[k]
            logger.info("%s sample %-19s: %.4f",
                        config['evaluation']['metric'], sample, float(metric_dict))
            metrics[sample] = float(metric_dict)
    else:
        for metric_dict, sample in zip(metric_dicts, samples):
            if metric_dict is None:
                continue
            metric_dict = metric_dict['confusion_matrix']
            metric = 0.0
            if returnIoUavg == "full":
                returnIoUavg = '123456789'
            for t in range(1, 10):
                if str(t) in returnIoUavg:
                    metric += metric_dict['th_0_' + str(t)]['AP']
            metric /= len(returnIoUavg)
            logger.info("confusion_matrix.avg.th_0_%s.AP sample %-19s: %.4f",
                        returnIoUavg, sample, float(metric))
            metrics[sample] = float(metric)

    if 'summary' in config['evaluation'].keys():
        tmp_metric_dicts = []
        tmp_samples = []
        for metric_dict, sample in zip(metric_dicts, samples):
            if "eft3RW10035L1_0125072" not in sample and \
               "unc54L1_0123072" not in sample:
                tmp_metric_dicts.append(metric_dict)
                tmp_samples.append(sample)
        eval_seg.summarize_metric_dict(
            tmp_metric_dicts,
            tmp_samples,
            config['evaluation']['summary'],
            os.path.join(output_folder, 'summary.csv')
        )

    if return_avg:
        return np.mean(list(metrics.values()))
    else:
        return metrics, metrics_full


def visualize():
    raise NotImplementedError


@time_func
def cross_validate(args, config, data, train_folder, test_folder):
    # data = config['data']['test_data']
    num_variations = 0
    # print(config['cross_validate'])
    for k, v in config['cross_validate'].items():
        if k == 'checkpoints':
            continue
        elif num_variations == 0:
            num_variations = len(v)
        else:
            assert num_variations == len(v), \
                'number of values for parameters has to be fixed'
    for checkpoint in config['cross_validate']['checkpoints']:
        pred_folder = os.path.join(test_folder, 'processed', str(checkpoint))
        checkpoint_file = get_checkpoint_file(checkpoint,
                                              config['model']['train_net_name'],
                                              train_folder)
        predict(args, config, config['model']['test_net_name'], data,
                checkpoint_file, test_folder, pred_folder)

    params = config['validation'].get('params', [])
    results = {}
    for checkpoint in config['cross_validate']['checkpoints']:
        for i in range(num_variations):
            param_set = {}
            for p in params:
                param_set[p] = config['cross_validate'][p][i] \
                               if p in config['cross_validate'] \
                                  else config['vote_instances'][p]
            params_str = [k + "_" + str(v).replace(".", "_")
                          for k, v in param_set.items()]
            inst_folder = os.path.join(test_folder, 'instanced',
                                       str(checkpoint), *params_str)
            os.makedirs(inst_folder, exist_ok=True)
            eval_folder = os.path.join(test_folder, 'evaluated',
                                       str(checkpoint), *params_str)
            os.makedirs(eval_folder, exist_ok=True)
            logger.info("labelling: %s", param_set)
            label(args, config, data, pred_folder, inst_folder,
                  param_set)
            metrics = evaluate(args, config, data, inst_folder, eval_folder,
                               return_avg=False)
            results[(checkpoint, *(param_set.values()))] = metrics

    samples = natsorted(get_list_samples(config, data, config['data']['input_format']))
    for k, v in results.items():
        assert len(v[0]) == len(samples)
        for s1, s2 in zip(v[0].keys(), samples):
            assert s1 == s2
    random.Random(42).shuffle(samples)
    samples_fold1 = set(samples[:len(samples)//2])
    samples_fold2 = set(samples[len(samples)//2:])
    results_fold1 = {}
    results_fold2 = {}
    for setup, result in results.items():
        acc = []
        for s in samples_fold1:
            acc.append(result[0][s])
        acc = np.mean(acc)
        results_fold1[setup] = acc

        acc = []
        for s in samples_fold2:
            acc.append(result[0][s])
        acc = np.mean(acc)
        results_fold2[setup] = acc

    best_setup_fold1 = max(results_fold1.items(), key=operator.itemgetter(1))[0]
    best_setup_fold2 = max(results_fold2.items(), key=operator.itemgetter(1))[0]

    acc1 = []
    for s in samples_fold2:
        acc1.append(results[best_setup_fold1][0][s])
    acc2 = []
    for s in samples_fold1:
        acc2.append(results[best_setup_fold2][0][s])

    acc = np.mean(acc1+acc2)
    acc1 = np.mean(acc1)
    acc2 = np.mean(acc2)

    logger.info("%s CROSS: %.4f [%.4f (%s), %.4f (%s)]",
                config['evaluation']['metric'], acc,
                acc1, best_setup_fold2,
                acc2, best_setup_fold1)
    print("%s CROSS: %.4f [%.4f (%s), %.4f (%s)]" % (
        config['evaluation']['metric'], acc,
        acc1, best_setup_fold2,
        acc2, best_setup_fold1))

    ap_ths = ["confusion_matrix.th_0_1.AP",
              "confusion_matrix.th_0_2.AP",
              "confusion_matrix.th_0_3.AP",
              "confusion_matrix.th_0_4.AP",
              "confusion_matrix.th_0_5.AP",
              "confusion_matrix.th_0_6.AP",
              "confusion_matrix.th_0_7.AP",
              "confusion_matrix.th_0_8.AP",
              "confusion_matrix.th_0_9.AP"]
    for ap_th in ap_ths:
        metric_dicts = results[best_setup_fold2][1]
        metrics = {}
        for sample, metric_dict in metric_dicts.items():
            if metric_dict is None:
                continue
            for k in ap_th.split('.'):
                metric_dict = metric_dict[k]
            metrics[sample] = float(metric_dict)
        metric = np.mean(list(metrics.values()))
        print("%s: %.4f" % (ap_th, metric))


def main():
    # parse command line arguments
    args = get_arguments()

    if not args.do:
        raise ValueError("Provide a task to do (-d/--do)")

    # get experiment name and create folder
    if args.expid is not None:
        if os.path.isdir(args.expid):
            base = args.expid
        else:
            base = os.path.join(args.root, args.expid)
    else:
        base = os.path.join(args.root, args.setup + '_' + \
                            datetime.now().strftime('%y%m%d_%H%M%S'))

    # create folder structure for experiment
    if args.debug_args:
        base = base.replace("experiments", "experimentsTmp")
    train_folder, val_folder, test_folder = create_folders(args, base)

    # read config file
    if args.config is None and args.expid is None:
        raise RuntimeError("No config file provided (-c/--config)")
    elif args.config is None:
        args.config = [os.path.join(base, 'config.toml')]
    try:
        config = {}
        for conf in args.config:
            config = merge_dicts(config, toml.load(conf))
    except:
        raise IOError('Could not read config file: {}! Please check!'.format(
            conf))
    config['base'] = base

    # set logging level
    logging.basicConfig(
        level=config['general']['logging'],
        handlers=[
            logging.FileHandler(os.path.join(base, "run.log"), mode='a'),
            logging.StreamHandler(sys.stdout)
        ])
    logger.info('attention: using config file %s', args.config)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        try:
            selectedGPU = util.selectGPU(
                quantity=config['training'].get('num_gpus', 1))
        except FileNotFoundError:
            selectedGPU = None
        if selectedGPU is None:
            logger.warning("no free GPU available!")
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                [str(i) for i in selectedGPU])
        logger.info("setting CUDA_VISIBLE_DEVICES to device {}".format(
            selectedGPU))
    else:
        logger.info("CUDA_VISIBILE_DEVICES already set, device {}".format(
            os.environ["CUDA_VISIBLE_DEVICES"]))

    # update config with command line values
    update_config(args, config)

    if args.trained_model:
        model_results = []
        for model in args.trained_model:
            with open(os.path.join(model, 'val', "results.json"), 'r') as f:
                results = json.load(f)
            entries = {}
            for r in results:
                entry = [r['checkpoint']]
                for p in sorted(r['params'].keys()):
                    entry.append(r['params'][p])
                entries[tuple(entry)] = r['accuracy']
            model_results.append(entries)
        results = pd.DataFrame(model_results)
        logger.info("Results:\n %s\n", results.transpose())
        logger.info("Results (mean):\n %s\n", results.mean())
        mean_results = dict(results.mean())
        best_params = max(mean_results, key=mean_results.get)

        config['training']['max_iterations'] = best_params[0] + 10
        for idx, p in enumerate(['checkpoint'] + sorted(r['params'].keys())):
            logger.info("Best parameters: %s = %s", p, best_params[idx])
            config['postprocessing'][p] = best_params[idx]
        assert config['training'].get('folds') is None, \
            'folds should not be set in training for final model'
        assert 'validate' not in args.do, \
            'no validation data for final model'
        assert 'validate_checkpoints' not in args.do, \
            'no validation data for final model'
        assert 'all' not in args.do, \
            'no validation for final model, specify tasks explicitly'

    # backup and copy config
    backup_and_copy_file(None, base, 'config.toml')
    with open(os.path.join(base, "config.toml"), 'w') as f:
        toml.dump(config, f)
    if args.do_detection:
        config['evaluation']['metric'] = 'confusion_matrix.hoefener.AP'
        config['evaluation']['use_linear_sum_assignment'] = False
        config['evaluation']['detection'] = "hoefener"
        config['evaluation']['summary'] = [ "confusion_matrix.hoefener.AP",]
    if args.debug_args:
        setDebugValuesForConfig(config)
    logger.info('used config: %s', config)

    # create network
    if 'all' in args.do or 'mknet' in args.do:
        mknet(args, config, train_folder, test_folder)

    # train network
    if 'all' in args.do or 'train' in args.do:
        train(args, config, train_folder)

    # determine which checkpoint to use
    checkpoint = None
    if any(i in args.do for i in ['all', 'validate', 'predict', 'label',
                                  'evaluate']):
        if args.checkpoint is not None:
            checkpoint = int(args.checkpoint)
            checkpoint_path = os.path.join(
                train_folder, config['model']['train_net_name'], '_checkpoint_',
                str(checkpoint))

        elif args.test_checkpoint == 'last':
            checkpoint_file = os.path.join(train_folder, 'checkpoint')
            assert os.path.isfile(checkpoint_file), \
                "checkpoint file not found: %s" % checkpoint_file
            with open(checkpoint_file) as f:
                d = dict(
                    x.rstrip().replace('"', '').replace(':', '').split(None, 1)
                    for x in f)
            checkpoint_path = d['model_checkpoint_path']
            try:
                checkpoint = int(checkpoint_path.split('_')[-1])
            except ValueError:
                logger.error('Could not convert checkpoint to int.')
                raise

        if checkpoint is None and \
           args.test_checkpoint != 'best' and \
           any(i in args.do for i in ['validate', 'predict', 'decode',
                                      'label', 'evaluate', 'infer']):
            raise ValueError(
                'Please provide a checkpoint (--checkpoint/--test-checkpoint)')

    params = None
    # validation:
    # validate all checkpoints
    if ([do for do in args.do if do in ['all', 'predict', 'label',
                                       'postprocess', 'evaluate']]\
        and args.test_checkpoint == 'best') \
        or 'validate_checkpoints' in args.do:
        data, output_folder = select_validation_data(config, train_folder,
                                                     val_folder)
        if config['validation'].get('checkpoints'):
            checkpoints = config['validation']['checkpoints']
        else:
            checkpoints = get_checkpoint_list(config['model']['train_net_name'],
                                              train_folder)
        logger.info("validating all checkpoints")
        assert len(checkpoints) > 0, "no checkpoints found!"
        checkpoint, params = validate_checkpoints(args, config, data,
                                                  checkpoints,
                                                  train_folder, test_folder,
                                                  output_folder)
    # validate single checkpoint
    else:
        if 'validate' in args.do:
            if checkpoint is None:
                raise RuntimeError("checkpoint must be set but is None")
            data, output_folder = select_validation_data(config, train_folder,
                                                         val_folder)
            _ = validate_checkpoints(args, config, data, [checkpoint],
                                     train_folder, test_folder, output_folder)

    if [do for do in args.do if do in ['all', 'predict', 'label',
                                       'postprocess', 'evaluate']]:
        if checkpoint is None:
            raise RuntimeError("checkpoint must be set but is None")

        if args.test_checkpoint != 'best':
            params = get_postprocessing_params(
                None,
                config['validation'].get('params', []),
                config['postprocessing'])
            for k,v in params.items():
                params[k] = v[0]

        params_str = [k + "_" + str(v).replace(".", "_")
                      for k, v in params.items()]
        pred_folder = os.path.join(test_folder, 'processed', str(checkpoint))
        inst_folder = os.path.join(test_folder, 'instanced', str(checkpoint),
                                   *params_str)
        detect_str = "detection" \
                     if config['evaluation'].get("do_detection", False) or \
                        args.do_detection \
                     else ""
        eval_folder = os.path.join(test_folder, 'evaluated',
                                   detect_str,
                                   str(checkpoint), *params_str)

    # predict test set
    if 'all' in args.do or 'predict' in args.do:

        # assume checkpoint has been determined already
        os.makedirs(pred_folder, exist_ok=True)

        checkpoint_file = get_checkpoint_file(
            checkpoint, config['model']['train_net_name'], train_folder)
        logger.info("predicting checkpoint %d", checkpoint)
        predict(args, config, config['model']['test_net_name'],
                config['data']['test_data'], checkpoint_file,
                test_folder, pred_folder)

    if 'all' in args.do or 'label' in args.do:
        os.makedirs(inst_folder, exist_ok=True)
        logger.info("labelling checkpoint %d", checkpoint)
        label(args, config, config['data']['test_data'], pred_folder,
              inst_folder, params)

    if 'all' in args.do or 'postprocess' in args.do:
        print('postprocess')

    if 'all' in args.do or 'evaluate' in args.do:
        os.makedirs(eval_folder, exist_ok=True)
        logger.info("evaluating checkpoint %d", checkpoint)
        metric = evaluate(args, config, config['data']['test_data'],
                          inst_folder, eval_folder)
        logger.info("%s TEST checkpoint %d: %.4f (%s)",
                    config['evaluation']['metric'], checkpoint, metric,
                    params)

    if 'all' in args.do or 'visualize' in args.do:
        visualize()

    if 'cross_validate' in args.do:
        cross_validate(args, config, config['data']['val_data'], train_folder, val_folder)

if __name__ == "__main__":
    main()
