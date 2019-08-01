import argparse
from datetime import datetime
from glob import glob
import fnmatch
import functools
import importlib
import logging
try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)
import os
import shutil
import sys
import time

import h5py
from joblib import Parallel, delayed
import numpy as np
import toml

from auxcpvloss import util
from evaluateInstanceSegmentation import evaluate_file


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


def time_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = datetime.now()
        ret = func(*args, **kwargs)
        logger.info('time %s: %s', func.__name__, str(datetime.now() - t0))
        return ret
    return wrapper


logger = logging.getLogger(__name__)


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', action='append',
                        help=('Configuration files to use. For defaults, '
                              'see `config/default.toml`.'))
    parser.add_argument('-a', '--app', dest='app', required=True,
                        help=('Application to use. Choose out of cityscapes, '
                              'flylight, kaggle, etc.'))
    parser.add_argument('-r', '--root', dest='root', default=None,
                        help='Experiment folder to store results.',
                        required=True)
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
                                 'validate_checkpoints',
                                 'validate',
                                 'label',
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

    args = parser.parse_args()

    return args


def create_folders(args, expname):
    # create experiment folder
    filebase = os.path.join(args.root, expname)
    os.makedirs(filebase, exist_ok=True)

    setup = os.path.join(args.app, '02_setups', args.setup)
    backup_and_copy_file(setup, filebase, 'train.py')
    backup_and_copy_file(setup, filebase, 'mknet.py')
    backup_and_copy_file(setup, filebase, 'predict.py')
    backup_and_copy_file(setup, filebase, 'label.py')

    # create train folders
    train_folder = os.path.join(filebase, 'train')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'snapshots'), exist_ok=True)

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

    return filebase, train_folder, val_folder, test_folder


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

def setDebugValuesForConfig(config):
    config['training']['max_iterations'] = 10
    config['training']['checkpoints'] = 10
    config['training']['snapshots'] = 10
    config['training']['profiling'] = 10
    # config['training']['num_workers'] = 1
    # config['training']['cache_size'] = 0


@time_func
def mknet(args, config, train_folder, test_folder):
    mknet = importlib.import_module(
        args.app + '.02_setups.' + args.setup + '.mknet')

    mknet.mk_net(name=config['model']['train_net_name'],
                 input_shape=config['model']['train_input_shape'],
                 output_folder=train_folder,
                 **config['model'], **config['optimizer'],
                 debug=config['general']['debug'])
    mknet.mk_net(name=config['model']['test_net_name'],
                 input_shape=config['model']['test_input_shape'],
                 output_folder=test_folder,
                 **config['model'], **config['optimizer'],
                 debug=config['general']['debug'])


@time_func
def train(args, config, train_folder):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError("no free GPU available!")
    child_pid = os.fork()
    if child_pid == 0:
        data_files = get_list_train_files(config)
        train = importlib.import_module(
            args.app + '.02_setups.' + args.setup + '.train')

        train.train_until(name=config['model']['train_net_name'],
                          max_iteration=config['training']['max_iterations'],
                          output_folder=train_folder,
                          data_files=data_files,
                          voxel_size=config['data']['voxel_size'],
                          input_format=config['data']['input_format'],
                          **config['training'],
                          **config['preprocessing'])
        os._exit(0)

    else:
        _, status = os.waitpid(child_pid, 0)
        exitcode = os.WEXITSTATUS(status)
        if not os.WIFEXITED(status) or exitcode != 0:
            raise RuntimeError("training failed, check exception in child process")


def get_list_train_files(config):
    data = config['data']['train_data']
    if os.path.isfile(data):
        files = [data]
    elif os.path.isdir(data):
        if 'folds' in config['training']:
            files = []
            for c in config['training']['folds']:
                files += glob(os.path.join(
                    data + "_fold" + c, "*." + config['data']['input_format']))
        else:
            files = glob(os.path.join(data,
                                      "*." + config['data']['input_format']))
    else:
        raise ValueError(
            "please provide file or directory for data/train_data", data)
    return files


def get_list_samples(config, data, file_format):

    # read data
    if os.path.isfile(data):
        if file_format == ".hdf":
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


def predict_sample(args, config, name, data, sample, checkpoint, input_folder,
                   output_folder):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError("no free GPU available!")

    predict = importlib.import_module(
        args.app + '.02_setups.' + args.setup + '.predict')

    predict.predict(name=name, sample=sample, checkpoint=checkpoint,
                    data_folder=data, input_folder=input_folder,
                    output_folder=output_folder,
                    voxel_size=config['data']['voxel_size'],
                    input_format=config['data']['input_format'],
                    **config['preprocessing'],
                    **config['model'],
                    **config['prediction'])


@time_func
def predict(args, config, name, data, checkpoint, test_folder, output_folder):

    samples = get_list_samples(config, data, config['data']['input_format'])

    for sample in samples:
        if not config['general']['overwrite'] and \
           os.path.exists(os.path.join(
               output_folder,
               sample + '.' + config['prediction']['output_format'])):
            logger.info('Skipping prediction for %s. Already exists!', sample)
            continue

        # forking for each prediction to terminate tensorflow server on gpu
        # TODO: check if there is a nicer, cleaner way
        child_pid = os.fork()
        if child_pid == 0:
            predict_sample(args, config, name, data, sample, checkpoint,
                           test_folder, output_folder)
            os._exit(0)
        else:
            _, status = os.waitpid(child_pid, 0)
            exitcode = os.WEXITSTATUS(status)
            if not os.WIFEXITED(status) or exitcode != 0:
                raise RuntimeError("prediction failed, check exception in child process")


def get_checkpoint_file(iteration, name, train_folder):
    return os.path.join(train_folder, name + '_checkpoint_%d' % iteration)


def get_checkpoint_list(name, train_folder):
    checkpoints = glob(
        os.path.join(train_folder, name + '_checkpoint_*.index'))
    return [int(os.path.splitext(os.path.basename(cp))[0].split("_")[-1])
            for cp in checkpoints]


def select_validation_data(config, train_folder, val_folder):
    if config['data'].get('validate_on_train'):
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
def validate_checkpoint(args, config, data, checkpoint, train_folder,
                        test_folder, output_folder):
    logger.info("validating checkpoint %d", checkpoint)
    # create test iteration folders
    pred_folder = os.path.join(output_folder, 'processed', str(checkpoint))
    inst_folder = os.path.join(output_folder, 'instanced', str(checkpoint))
    eval_folder = os.path.join(output_folder, 'evaluated', str(checkpoint))

    for out in [pred_folder, inst_folder, eval_folder]:
        os.makedirs(out, exist_ok=True)

    # predict val data
    checkpoint_file = get_checkpoint_file(checkpoint,
                                          config['model']['train_net_name'],
                                          train_folder)
    logger.info("predicting checkpoint %d", checkpoint)
    predict(args, config, config['model']['test_net_name'], data,
            checkpoint_file, test_folder, pred_folder)

    # label
    logger.info("labelling checkpoint %d", checkpoint)
    label(args, config, data, pred_folder, inst_folder)

    # evaluate
    logger.info("evaluating checkpoint %d", checkpoint)
    acc = evaluate(args, config, data, inst_folder, eval_folder)
    logger.info("AP checkpoint %6d: %.4f", checkpoint, acc)

    return acc


def validate_checkpoints(args, config, data, checkpoints, train_folder,
                         test_folder, output_folder):
    # validate all checkpoints and return best one
    accs = []
    ckpts = []
    for checkpoint in checkpoints:
        acc = validate_checkpoint(args, config, data, checkpoint,
                                  train_folder, test_folder, output_folder)
        ckpts.append(checkpoint)
        accs.append(acc)

    for ch, acc in zip(checkpoints, accs):
        logger.info("AP checkpoint %6d: %.4f", ch, acc)

    for ch, acc in zip(checkpoints, accs):
        logger.info("AP %d: %f", ch, acc)

    if config['general']['debug'] and None in accs:
        logger.error("None in checkpoint found: %s (continuing with last)",
                     tuple(accs))
        best_checkpoint = ckpts[-1]
    else:
        best_checkpoint = ckpts[np.argmax(accs)]
    logger.info('best checkpoint: %d', best_checkpoint)
    return best_checkpoint


def label_sample(args, config, data, pred_folder, output_folder, sample):
    label = importlib.import_module(
        args.app + '.02_setups.' + args.setup + '.label')

    # TODO: logging doesn't work in parallel case
    if not config['general']['overwrite'] and \
       os.path.exists(os.path.join(
           output_folder,
           sample + '.' + config['postprocessing']['watershed']['output_format'])):
        logger.info('Skipping labelling for %s. Already exists!', sample)
        return

    if config['postprocessing']['watershed']['include_gt']:
        gt = os.path.join(data, sample + "." + config['data']['input_format'])
    else:
        gt = None
    label.label(sample=sample, gt=gt, pred_folder=pred_folder,
                output_folder=output_folder,
                pred_format=config['prediction']['output_format'],
                gt_format=config['data']['input_format'],
                gt_key=config['data']['gt_key'],
                **config['postprocessing']['watershed'])


@time_func
def label(args, config, data, pred_folder, output_folder):
    samples = get_list_samples(config, pred_folder,
                               config['prediction']['output_format'])
    num_workers = config['postprocessing']['watershed'].get("num_workers", 1)
    if num_workers > 1:
        Parallel(n_jobs=num_workers, backend='multiprocessing', verbose=1) \
            (delayed(label_sample)(args, config, data, pred_folder,
                                   output_folder, s) for s in samples)
    else:
        for sample in samples:
            label_sample(args, config, data, sample, pred_folder,
                         output_folder)


def evaluate_sample(config, data, sample, inst_folder, output_folder,
                    file_format):
    if os.path.isfile(data):
        gt_path = data
        gt_key = sample + "/gt"
    else:
        gt_path = os.path.join(data,
                               sample + "."+config['data']['input_format'])
        gt_key = config['data']['gt_key']

    sample_path = os.path.join(inst_folder, sample + "." + file_format)
    return evaluate_file(sample_path, gt_path,
                         res_key=config['evaluation']['res_key'],
                         gt_key=gt_key,
                         out_dir=output_folder, suffix="",
                         debug=config['general']['debug'])


@time_func
def evaluate(args, config, data, inst_folder, output_folder):
    file_format = config['postprocessing']['watershed']['output_format']
    samples = get_list_samples(config, inst_folder, file_format)

    # TODO: sorry, need to add zarr to evaluation script
    num_workers = config['evaluation'].get("num_workers", 1)
    if num_workers > 1:
        metric_dicts = Parallel(n_jobs=num_workers, backend='multiprocessing',
                                verbose=0) \
            (delayed(evaluate_sample)(config, data, s, inst_folder,
                                      output_folder, file_format)
             for s in samples)
    else:
        metric_dicts = []
        for sample in samples:
            metric_dict = evaluate_sample(config, data, sample, inst_folder,
                                          output_folder, file_format)
            if metric_dict is None:
                continue
            metric_dicts.append(metric_dict)

    accs = []
    for metric_dict, sample in zip(metric_dicts, samples):
        if metric_dict is None:
            continue
        for k in config['evaluation']['metric'].split('/'):
            metric_dict = metric_dict[k]
        logger.info("AP sample %-19s: %.4f", sample, metric_dict)
        accs.append(metric_dict)

    return np.mean(accs)


def visualize():
    print('visualize')


def main():
    # parse command line arguments
    args = get_arguments()

    if not args.do:
        raise ValueError("Provide a task to do (-d/--do)")

    # get experiment name
    if args.setup is not None:
        if args.expid is not None:
            expname = args.setup + '_' + args.expid
        else:
            expname = args.setup + '_' + datetime.now().strftime('%y%m%d_%H%M%S')

    # create folder structure for experiment
    base, train_folder, val_folder, test_folder = create_folders(args, expname)

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

    # set logging level
    logging.basicConfig(
        level=config['general']['logging'],
        handlers=[
            logging.FileHandler(os.path.join(base, "run.log"), mode='a'),
            logging.StreamHandler(sys.stdout)
        ])
    logger.info('attention: using config file %s', args.config)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        selectedGPU = util.selectGPU()
        if selectedGPU is None:
            logger.warning("no free GPU available!")
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(selectedGPU)
        logger.info("setting CUDA_VISIBLE_DEVICES to device {}".format(
            selectedGPU))
    else:
        logger.info("CUDA_VISIBILE_DEVICES already set, device {}".format(
            os.environ["CUDA_VISIBLE_DEVICES"]))

    # update config with command line values
    update_config(args, config)
    backup_and_copy_file(None, base, 'config.toml')
    with open(os.path.join(base, "config.toml"), 'w') as f:
        toml.dump(config, f)
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
            with open(os.path.join(train_folder, 'checkpoint')) as f:
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
           any(i in args.do for i in ['validate', 'predict', 'evaluate']):
            raise ValueError(
                'Please provide a checkpoint (--checkpoint/--test_checkpoint)')

    # validation:
    # validate all checkpoints
    if ('all' in args.do and args.test_checkpoint == 'best') \
            or 'validate_checkpoints' in args.do:
        data, output_folder = select_validation_data(config, train_folder,
                                                     val_folder)
        checkpoints = get_checkpoint_list(config['model']['train_net_name'],
                                          train_folder)
        logger.info("validating all checkpoints")
        checkpoint = validate_checkpoints(args, config, data, checkpoints,
                                          train_folder, test_folder,
                                          output_folder)
    # validate single checkpoint
    elif 'validate' in args.do:
        data, output_folder = select_validation_data(config, train_folder,
                                                     val_folder)
        _ = validate_checkpoint(args, config, data, checkpoint, train_folder,
                                test_folder, output_folder)

    if checkpoint is not None:
        pred_folder = os.path.join(test_folder, 'processed', str(checkpoint))
        inst_folder = os.path.join(test_folder, 'instanced', str(checkpoint))
        eval_folder = os.path.join(test_folder, 'evaluated', str(checkpoint))

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
              inst_folder)

    if 'all' in args.do or 'postprocess' in args.do:
        print('postprocess')

    if 'all' in args.do or 'evaluate' in args.do:
        os.makedirs(eval_folder, exist_ok=True)
        logger.info("evaluating checkpoint %d", checkpoint)
        acc = evaluate(args, config, config['data']['test_data'], inst_folder,
                       eval_folder)
        logger.info("AP TEST checkpoint %d: %.4f", checkpoint, acc)

    if 'all' in args.do or 'visualize' in args.do:
        visualize()


if __name__ == "__main__":
    main()
