import argparse
import glob
import os

import h5py
from joblib import Parallel, delayed
import numpy as np
import scipy.ndimage
import scipy.stats
import skimage.io as io
import tifffile
import zarr


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


def get_and_write_points(filename, out_fn, shape, padding=0):
    gt_centers = np.zeros(shape, dtype=np.float32)
    with open(filename, 'r') as in_fl:
        out_fl = open(out_fn, 'w')
        # one line per instance, extract center points
        for ln in in_fl:
            arr = ln.strip().split(" ")
            idx = float(arr[0])
            x = float(arr[2])+padding
            y = float(arr[3])+padding
            z = float(arr[4])+padding
            gt_centers[int(z), int(y), int(x)] = 1.0
            out_fl.write("{}, {}, {}, {}\n".format(z, y, x, idx))
    return gt_centers

# https://github.com/CSBDeep/CSBDeep/blob/master/csbdeep/utils/utils.py
def normalize_percentile(x, pmin=3, pmax=99.8, axis=None, clip=False,
                         eps=1e-20, dtype=np.float32):
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    # print("min/max", mi, ma, np.min(x), np.max(x))
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
    # print("{} before norm {}: min {}, max {}, mean {}, std {}, median {}".format(
    #     sample, args.normalize, np.min(raw), np.max(raw), np.mean(raw),
        # np.std(raw), np.median(raw)))

    if args.normalize == "minmax":
        raw = normalize_min_max(raw, args.raw_min, args.raw_max)
    elif args.normalize == "percentile":
        raw = normalize_percentile(raw, args.raw_min, args.raw_max)

    # print("{} after norm {}:  min {}, max {}, mean {}, std {}, median {}".format(
    #     sample, args.normalize, np.min(raw), np.max(raw), np.mean(raw),
    #     np.std(raw), np.median(raw)))
    return raw


def preprocess(args, raw, sample):
    # print("{} before preproc {}: skew {}".format(
    #     sample, args.preprocess, scipy.stats.skew(raw.ravel())))
    if args.preprocess is None or args.preprocess == "no":
        pass
    elif args.preprocess == "square":
        raw = np.square(raw)
    elif args.preprocess == "cuberoot":
        raw = np.cbrt(raw)
    # print("{} after preproc {}:  skew {}".format(
    #     sample, args.preprocess, scipy.stats.skew(raw.ravel())))
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
    parser.add_argument('--out-format', dest='out_format', default="hdf",
                        help='format of output files')
    parser.add_argument('-p', '--parallel', default=1, type=int)
    parser.add_argument('--raw-min', dest='raw_min', type=float)
    parser.add_argument('--raw-max', dest='raw_max', type=float)
    parser.add_argument('--scale-sdt', dest='scale_sdt', type=float, default=-9)
    parser.add_argument('--sigma', type=float, default=2)
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

    # if "C50F7_5SD1583L1_0615081" not in sample:
    #     return
    try:
        raw_fn = os.path.join(args.in_dir, "imagesAsMhdRaw", sample + ".mhd")
        raw = load_array(raw_fn).astype(np.float32)
    except:
        raw_fn = os.path.join(args.in_dir, "imagesAsTiff", sample + ".tiff")
        raw = load_array(raw_fn).astype(np.float32)
    raw = preprocess(args, raw, sample)
    raw = normalize(args, raw, sample)
    raw = pad(args, raw, 'mode')

    labels_fn = os.path.join(args.in_dir, "groundTruthInstanceSeg",
                             sample + ".ano.curated.tiff")
    print(raw_fn, labels_fn)

    gt_labels = load_array(labels_fn).astype(np.uint16)
    # print("labels cnt {}, min {}, max {}".format(len(np.unique(gt_labels)), np.min(gt_labels), np.max(gt_labels)))
    gt_labels = pad(args, gt_labels, 'constant')
    # gt_labels = (gt_labels % 255).astype(np.uint8)

    gt_labels_tmp = np.zeros((gt_labels.shape[0]+1,
                              gt_labels.shape[1]+1,
                              gt_labels.shape[2]+1), dtype=np.uint16)
    gt_labels_tmp[1:,1:,1:] = gt_labels
    gt_affs = np.zeros((3,) + gt_labels.shape, dtype=np.uint8)
    gt_affs[0,...] = gt_labels==gt_labels_tmp[0:-1,1:,1:]
    gt_affs[0,...][gt_labels == 0] = 0
    gt_affs[1,...] = gt_labels==gt_labels_tmp[1:,0:-1,1:]
    gt_affs[1,...][gt_labels == 0] = 0
    gt_affs[2,...] = gt_labels==gt_labels_tmp[1:,1:,0:-1]
    gt_affs[2,...][gt_labels == 0] = 0

    gt_fgbg = np.zeros(gt_labels.shape, dtype=np.uint8)
    gt_fgbg[gt_labels > 0] = 1

    gt_threeclass = np.zeros(gt_labels.shape, dtype=np.uint8)
    struct = scipy.ndimage.generate_binary_structure(3, 3)
    for label in np.unique(gt_labels):
        if label == 0:
            continue

        label_mask = gt_labels==label
        eroded_label_mask = scipy.ndimage.binary_erosion(label_mask,
                                                         iterations=1,
                                                         structure=struct,
                                                         border_value=1)
        boundary = np.logical_xor(label_mask, eroded_label_mask)
        gt_threeclass[boundary] = 2
        gt_threeclass[eroded_label_mask] = 1

    gt_boundary = np.copy(gt_threeclass)
    gt_boundary[gt_threeclass != 2] = 1
    gt_boundary[gt_threeclass == 2] = 0

    gt_edt = scipy.ndimage.distance_transform_edt(gt_boundary)
    gt_sdt = np.copy(gt_edt)
    for label in np.unique(gt_labels):
        if label == 0:
            continue

        label_mask = gt_labels==label
        gt_sdt[label_mask] *= -1
    gt_tanh =  np.tanh(1./abs(args.scale_sdt) * gt_sdt)



    cp_fn = os.path.join(args.in_dir, "groundTruthInstanceSeg",
                         sample + ".ano.curated.txt")
    gt_centers = get_and_write_points(cp_fn, out_fn + ".csv", gt_labels.shape,
                                      padding=args.padding)
    gt_gaussian = scipy.ndimage.filters.gaussian_filter(gt_centers, args.sigma,
                                                        mode='constant')
    max_value = np.max(gt_gaussian)
    if max_value > 0:
        gt_gaussian /= max_value
    gt_gaussian = gt_gaussian.astype(np.float32)

    # tmp = np.zeros((3,) + gt_labels.shape, dtype=np.float32)
    # tmp[0,...] = gt_gaussian
    # tmp[1,...] = gt_gaussian
    # tmp[2,...] = gt_gaussian
    # tmp[0,...][gt_centers > 0.5] = 1.0
    # tmp[1,...][gt_centers > 0.5] = 0.0
    # tmp[2,...][gt_centers > 0.5] = 0.0
    # tmp *= 255
    # tmp = tmp.astype(np.uint8)
    # with h5py.File(out_fn + "_tmp.hdf", 'w') as f:
    #     f.create_dataset(
    #         'volumes/tmp',
    #         data=tmp,
    #         chunks=(3, 140, 140, 20),
    #         compression='gzip')

    if gt_labels.shape[0] != 1:
        gt_labels = np.expand_dims(gt_labels, 0)
        gt_tanh = np.expand_dims(gt_tanh, 0)
        gt_threeclass = np.expand_dims(gt_threeclass, 0)
        gt_fgbg = np.expand_dims(gt_fgbg, 0)
        gt_centers = np.expand_dims(gt_centers, 0)
        gt_gaussian = np.expand_dims(gt_gaussian, 0)

    if args.out_format == "hdf":
        f = h5py.File(out_fn + '.hdf', 'w')
    elif args.out_format == "zarr":
        f = zarr.open(out_fn + '.zarr', 'w')

    f.create_dataset(
        'volumes/raw',
        data=raw,
        chunks=(140, 140, 20),
        compression='gzip')
    f.create_dataset(
        'volumes/gt_labels',
        data=gt_labels,
        chunks=(1, 140, 140, 20),
        compression='gzip')
    f.create_dataset(
        'volumes/gt_fgbg',
        data=gt_fgbg,
        chunks=(1, 140, 140, 20),
        compression='gzip')
    f.create_dataset(
        'volumes/gt_tanh',
        data = gt_tanh,
        chunks=(1, 140, 140, 20),
        compression='gzip')
    f.create_dataset(
        'volumes/gt_threeclass',
        data = gt_threeclass,
        chunks=(1, 140, 140, 20),
        compression='gzip')
    f.create_dataset(
        'volumes/gt_affs',
        data = gt_affs,
        chunks=(3, 140, 140, 20),
        compression='gzip')
    f.create_dataset(
        'volumes/gt_centers',
        data = gt_centers,
        chunks=(1, 140, 140, 20),
        compression='gzip')
    f.create_dataset(
        'volumes/gt_gaussian',
        data = gt_gaussian,
        chunks=(1, 140, 140, 20),
        compression='gzip')

    for dataset in ['volumes/raw',
                    'volumes/gt_labels',
                    'volumes/gt_tanh',
                    'volumes/gt_threeclass',
                    'volumes/gt_affs',
                    'volumes/gt_centers',
                    'volumes/gt_gaussian',
                    'volumes/gt_fgbg']:
        f[dataset].attrs['offset'] = (0, 0, 0)
        f[dataset].attrs['resolution'] = (1, 1, 1)

    if args.out_format == "hdf":
        f.close()

if __name__ == "__main__":
    main()
