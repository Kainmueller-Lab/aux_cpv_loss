import argparse
from datetime import datetime
from glob import glob
import logging
import os
import importlib

import numpy as np
import h5py
import toml

# from auxcpvloss import util
from evaluateInstanceSegmentation import evaluate_files


logger = logging.getLogger(__name__)


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', dest='config', default=None,
                        help=('Configuration files to use. For defaults, '
                              'see `config/default.toml`.'))
    parser.add_argument('-a', '--app', dest='app', default=None,
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
    parser.add_argument('--data-format', dest='data_format',
                        choices=['hdf', 'zarr', 'n5', 'tif'],
                        help='File format of dataset.')
    parser.add_argument('--train-data', dest='train_data', default=None,
                        help='Train dataset to use.')
    parser.add_argument('--val-data', dest='val_data', default=None,
                        help='Validation dataset to use.')
    parser.add_argument('--test-data', dest='test_data', default=None,
                        help='Test dataset to use.')

    args = parser.parse_args()

    return args


def create_folders(args, expname):
    # create experiment folder
    filebase = os.path.join(args.root, expname)
    os.makedirs(filebase, exist_ok=True)

    # create train folders
    train_folder = os.path.join(filebase, 'train')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'snapshots'), exist_ok=True)

    # create val folders
    if 'validate' in args.do or 'validate_checkpoints' in args.do:
        val_folder = os.path.join(filebase, 'val')
        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(os.path.join(val_folder, 'processed'), exist_ok=True)
        os.makedirs(os.path.join(val_folder, 'instanced'), exist_ok=True)
    else:
        val_folder = None

    # create test folders
    test_folder = os.path.join(filebase, 'test')
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'instanced'), exist_ok=True)

    return filebase, train_folder, val_folder, test_folder


def get_data(args, config):
    # get train dataset

    if args.train_data is not None:
        train_data = args.train_data
        config['data']['train_data'] = train_data
    elif 'data' in config and 'train_data' in config['data']:
        train_data = config['data']['train_data']

    # get validation dataset
    if args.val_data is not None:
        val_data = args.val_data
        config['data']['val_data'] = val_data
    elif 'data' in config and 'val_data' in config['data']:
        val_data = config['data']['val_data']

    # get test dataset
    if args.test_data is not None:
        test_data = args.test_data
        config['data']['test_data'] = test_data
    elif 'data' in config and 'test_data' in config['data']:
        test_data = config['data']['test_data']

    if args.data_format is not None:
        config['data']['data_format'] = args.data_format

    return train_data, val_data, test_data


def mknet(args, config, train_folder):

    app = args.app + "." if args.app is not None else ""
    mknet = importlib.import_module(
        app + '02_setups.' + args.setup + '.mknet')

    mknet.mk_net('train_net', output_folder=train_folder,
                 input_shape=config['model']['train_input_shape'],
                 **config['model'], **config['optimizer'])
    mknet.mk_net('test_net', output_folder=train_folder,
                 input_shape=config['model']['test_input_shape'],
                 **config['model'], **config['optimizer'])


def train(args, config, train_folder):

    child_pid = os.fork()
    if child_pid == 0:
        data_files = get_list_files(config)
        app = args.app + "." if args.app is not None else ""
        train = importlib.import_module(
            app + '02_setups.' + args.setup + '.train')

        train.train_until(config['model']['train_net_name'],
                          max_iteration=config['training']['iteration'],
                          output_folder=train_folder,
                          data_files=data_files,
                          **config['training'],
                          **config['preprocessing'])
        os._exit(0)

    else:
        os.waitpid(child_pid, 0)


def get_list_files(config):
    data = config['data']['train_data']
    if os.path.isfile(data):
        files = [data]
    elif os.path.isdir(data):
        if config['training']['folds'] is not None:
            files = []
            for c in config['training']['folds']:
                files += glob(config['data']['train_data'] + "_fold" + c +
                              "*." + config['data']['data_format'])
    else:
        raise ValueError(
            "please provide file or directory for data/train_data", data)
    return files


def read_samples(config, data):

    # read data
    if os.path.isfile(data):
        if data.endswith(".hdf"):
            inf = h5py.File(data, 'r')
            samples = [k for k in inf]
            inf.close()
    elif os.path.isdir(data):
        samples = glob(os.path.join(data, "*."+config['data']['data_format']))
    else:
        raise NotImplementedError

    return samples


def predict_sample(args, config, data, sample, checkpoint, train_folder,
                   output_folder):

    app = args.app + "." if args.app is not None else ""
    predict = importlib.import_module(
        app + '02_setups.' + args.setup + '.predict')

    predict.predict_labels(sample, checkpoint, data, train_folder,
                           output_dir=output_folder,
                           **config['preprocessing'],
                           **config['model'],
                           **config['testing']
                           )


def predict(args, config, data, checkpoint, train_folder, output_folder):

    samples = read_samples(data)

    for sample in samples:
        print(os.path.join(output_folder, sample + '.zarr'))
        if os.path.exists(os.path.join(output_folder, sample + '.zarr')):
            print('Skipping prediction for ', sample, '. Already exists!')
            continue

        # forking for each prediction to terminate tensorflow server on gpu
        # todo: check if there is a nicer, cleaner way
        child_pid = os.fork()
        if child_pid == 0:
            predict_sample(args, config, data, sample, checkpoint,
                           train_folder, output_folder)
            os._exit(0)
        else:
            os.waitpid(child_pid, 0)


def validate_checkpoint(args, config, data, checkpoint, train_folder,
                        output_folder):

    # create test iteration folders
    pred_folder = os.path.join(
        output_folder, 'processed', str(checkpoint))
    inst_folder = os.path.join(
        output_folder, 'instanced', str(checkpoint))
    eval_folder = os.path.join(
        output_folder, 'evaluated', str(checkpoint))

    for out in [pred_folder, inst_folder, eval_folder]:
        os.makedirs(out, exist_ok=True)

    # predict val data
    predict(args, config, data, checkpoint, train_folder, pred_folder)

    # label
    label(args, config, data, pred_folder, inst_folder)

    # evaluate
    acc = evaluate(args, config, inst_folder, eval_folder)

    return acc


def validate_checkpoints(args, config, data, train_folder, output_folder):

    # check for available checkpoints in train folder
    checkpoints = glob(
        os.path.join(train_folder, 'train_net_checkpoint_*.index'))

    # validate all checkpoints and get best one
    accs = []
    ckpts = []

    for checkpoint_path in checkpoints:
        checkpoint = int(os.path.basename(checkpoint_path).split('.')[0].split(
            '_')[-1])
        acc = validate_checkpoint(args, config, data, checkpoint, train_folder,
                                  output_folder)
        ckpts.append(checkpoint)
        accs.append(acc)

    best_checkpoint = ckpts[np.argmax(accs)]
    print('best checkpoint: ', best_checkpoint)
    return best_checkpoint


def label(args, config, data, in_folder, out_folder):
    raise NotImplementedError


def evaluate(args, config, data, in_folder, out_folder):

    accs = []
    samples = read_samples(config, in_folder)
    # todo: sorry, need to add zarr to evaluation script
    for sample_path in samples:

        sample = os.path.basename(sample_path).split('.')[0]
        cargs = argparse.Namespace(
            res_file=sample_path,
            debug=False,
            gt_key=sample + '/gt',
            res_key='volumes/instances',
            suffix='',
            res_file_suffix=None,
            out_dir=out_folder,
        )
        metric_dict = evaluate_files(cargs, sample_path, data)
        for k in config['evaluate']['metric'].split('/'):
            metric_dict = metric_dict[k]
        accs.append(metric_dict)

    return np.mean(accs)


def visualize():
    print('visualize')


if __name__ == "__main__":

    # todo:

    # if "CUDA_VISIBLE_DEVICES" not in os.environ:
    #     from PatchPerPix.selectGPU import selectGPU
    #     selectedGPU = selectGPU()
    #     if selectedGPU is None:
    #         print("no free GPU available!")
    #     else:
    #         os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #         os.environ["CUDA_VISIBLE_DEVICES"] = selectedGPU
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    logging.basicConfig(level=logging.INFO)

    # parse command line arguments
    args = get_arguments()

    # read config file
    if args.config is not None:
        try:
            config = toml.load(args.config)
        except:
            print('Could not read config file! Please check!')
            raise
    else:
        config = {}

    # get experiment name
    if args.setup is not None:
        if args.expid is not None:
            expname = args.setup + '_' + args.expid
        else:
            expname = args.setup + '_' + datetime.now().strftime('%y%m%d_%H%M')

    # create folder structure for experiment
    _, train_folder, val_folder, test_folder = create_folders(args, expname)

    # get train, val, test data
    train_data, val_data, test_data = get_data(args, config)

    # create network
    if 'all' in args.do or 'mknet' in args.do:
        t0 = datetime.now()
        mknet(args, config, train_folder)
        logger.info('time mknet: %s', str(datetime.now() - t0))

    # train network
    if 'all' in args.do or 'train' in args.do:
        t0 = datetime.now()
        train(args, config, train_folder)
        logger.info('time train: %s', str(datetime.now() - t0))

    # determine which checkpoint to use
    if any(i in args.do for i in ['all', 'validate', 'predict', 'evaluate']):
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
                print('Could not convert checkpoint to int.')
                raise
        else:
            checkpoint = None

    # validate single checkpoint
    if 'validate' in args.do:
        t0 = datetime.now()
        validate_checkpoint(args, config, checkpoint, train_folder,
                            val_folder)
        logger.info('time validate: %s', str(datetime.now() - t0))

    # validate all checkpoints
    if ('all' in args.do and args.test_checkpoint == 'best') \
            or 'validate_checkpoints' in args.do:

        checkpoint = validate_checkpoints(
            args, config, val_data, train_folder, val_folder)

    pred_folder = os.path.join(test_folder, 'processed', str(checkpoint))
    inst_folder = os.path.join(test_folder, 'instanced', str(checkpoint))
    eval_folder = os.path.join(test_folder, 'evaluated', str(checkpoint))

    # predict test set
    if 'all' in args.do or 'predict' in args.do:

        # assume checkpoint has been determined already
        os.makedirs(pred_folder, exist_ok=True)

        t0 = datetime.now()
        predict(args, config, test_data, checkpoint, train_folder, pred_folder)
        logger.info('time predict: %s', str(datetime.now() - t0))

    if 'all' in args.do or 'postprocess' in args.do:
        print('postprocess')

    if 'all' in args.do or 'evaluate' in args.do:

        os.makedirs(eval_folder, exist_ok=True)
        t0 = datetime.now()
        evaluate(args, config, test_data, inst_folder, eval_folder)
        logger.info('time evaluate: %s', str(datetime.now() - t0))

    if 'all' in args.do or 'visualize' in args.do:
        visualize()
