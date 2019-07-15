import argparse
import glob
import os

import h5py
from joblib import Parallel, delayed
import numpy as np
import scipy.stats
import skimage.io as io
import tifffile


def load_array(filename):
    if filename.endswith(".tif") or \
       filename.endswith(".tiff") or \
       filename.endswith(".TIF") or \
       filename.endswith(".TIFF"):
        image = tifffile.imread(filename)
    elif filename.endswith(".mhd"):
        image = io.imread(filename, plugin="simpleitk")
    else:
        print("invalid file type")
        raise ValueError("invalid input file type", filename)

    return image


def get_and_write_points(filename, out_fn, padding=0):
    with open(filename, 'r') as in_fl:
        out_fl = open(out_fn, 'w')
        # one line per instance, extract center points
        for ln in in_fl:
            arr = ln.strip().split(" ")
            idx = float(arr[0])
            x = float(arr[2])+padding
            y = float(arr[3])+padding
            z = float(arr[4])+padding
            out_fl.write("{}, {}, {}, {}\n".format(z, y, x, idx))


# https://github.com/CSBDeep/CSBDeep/blob/master/csbdeep/utils/utils.py
def normalize_percentile(x, pmin=3, pmax=99.8, axis=None, clip=False,
                         eps=1e-20, dtype=np.float32):
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    print("min/max", mi, ma, np.min(x), np.max(x))
    return normalize_min_max(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_min_max(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if mi is None:
        mi = np.min(x)
    if ma is None:
        ma = np.max(x)
    if dtype is not None:
        x   = x.astype(dtype, copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x, 0, 1)

    return x


def normalize(args, raw, sample):
    print("{} before norm {}: min {}, max {}, mean {}, std {}, median {}".format(
        sample, args.normalize, np.max(raw), np.min(raw), np.mean(raw),
        np.std(raw), np.median(raw)))

    if args.normalize == "minmax":
        raw = normalize_min_max(raw, args.raw_min, args.raw_max)
    elif args.normalize == "percentile":
        raw = normalize_percentile(raw, args.raw_min, args.raw_max)

    print("{} after norm {}:  min {}, max {}, mean {}, std {}, median {}".format(
        sample, args.normalize, np.max(raw), np.min(raw), np.mean(raw),
        np.std(raw), np.median(raw)))
    return raw


def preprocess(args, raw, sample):
    print("{} before preproc {}: skew {}".format(
        sample, args.preprocess, scipy.stats.skew(raw.ravel())))
    if args.preprocess is None or args.preprocess == "no":
        pass
    elif args.preprocess == "square":
        raw = np.square(raw)
    elif args.preprocess == "cuberoot":
        raw = np.cbrt(raw)
    print("{} after preproc {}:  skew {}".format(
        sample, args.preprocess, scipy.stats.skew(raw.ravel())))
    return raw


def pad(args, array, mode):
    if args.padding != 0:
        array = np.pad(array,
                       ((args.padding, args.padding),
                        (args.padding, args.padding),
                        (args.padding, args.padding)),
                       mode)
    return array


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in-dir', dest='in_dir', required=True,
                        help='location of input files')
    parser.add_argument('-o', '--out-dir', dest='out_dir', required=True,
                        help='where to place output files')
    parser.add_argument('-p', '--parallel', default=1, type=int)
    parser.add_argument('--raw-min', dest='raw_min', type=int)
    parser.add_argument('--raw-max', dest='raw_max', type=int)
    parser.add_argument('--normalize', default='minmax',
                        choices=['minmax', 'percentile', 'meanstd'])
    parser.add_argument('--preprocess', default='no',
                        choices=['no', 'square', 'cuberoot'])
    parser.add_argument('--padding', default=0, type=int)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    return args


def main():
    args = get_arguments()

    files = map(lambda fn: fn.split("/")[-1].split(".")[0],
                glob.glob(os.path.join(
                    args.in_dir,'groundTruthInstanceSeg/*.ano.curated.tiff')))

    if args.parallel > 1:
        Parallel(n_jobs=args.parallel, verbose=1) \
            (delayed(work)(args, f) for f in files)
    else:
        for f in files:
            work(args, f)


def work(args, sample):
    print("Processing {}, {}".format(args.in_dir, sample))
    out_fn = os.path.join(args.out_dir, sample)

    raw_fn = os.path.join(args.in_dir, "imagesAsMhdRaw", sample + ".mhd")
    raw = load_array(raw_fn).astype(np.float32)
    raw = preprocess(args, raw, sample)
    raw = normalize(args, raw, sample)
    raw = pad(args, raw, 'mode')

    labels_fn = os.path.join(args.in_dir, "groundTruthInstanceSeg",
                             sample + ".ano.curated.tiff")
    labels = load_array(labels_fn).astype(np.uint16)
    labels = pad(args, labels, 'constant')
    if labels.shape[0] != 1:
        labels = np.expand_dims(labels, 0)


    cp_fn = os.path.join(args.in_dir, "groundTruthInstanceSeg",
                         sample + ".ano.curated.txt")
    get_and_write_points(cp_fn, out_fn + ".csv", padding=args.padding)

    fgbg = (1 * (labels > 0)).astype(np.uint8)

    with h5py.File(out_fn + '.hdf', 'w') as f:
        f.create_dataset(
            'volumes/raw',
            data=raw,
            compression='gzip')
        f.create_dataset(
            'volumes/gt_labels',
            data=labels,
            compression='gzip')
        f.create_dataset(
            'volumes/gt_fgbg',
            data=fgbg,
            compression='gzip')

        for dataset in ['volumes/raw',
                        'volumes/gt_labels',
                        'volumes/gt_fgbg']:
            f[dataset].attrs['offset'] = (0, 0, 0)
            f[dataset].attrs['resolution'] = (1, 1, 1)


if __name__ == "__main__":
    main()
