import argparse
import os
import sys
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest="in_dir", required=True)
    parser.add_argument('-o', dest="out_dir", required=True)
    parser.add_argument('-f', dest="out_format", default="hdf")
    args = parser.parse_args()

    trainD = os.path.join(args.out_dir, "train")
    trainD_f1 = os.path.join(args.out_dir, "train_fold1")
    trainD_f2 = os.path.join(args.out_dir, "train_fold2")
    trainD_f3 = os.path.join(args.out_dir, "train_fold3")
    trainD_f12 = os.path.join(args.out_dir, "train_folds12")
    trainD_f13 = os.path.join(args.out_dir, "train_folds13")
    trainD_f23 = os.path.join(args.out_dir, "train_folds23")
    testD = os.path.join(args.out_dir, "test")
    valD = os.path.join(args.out_dir, "val")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(trainD, exist_ok=True)
    os.makedirs(trainD_f1, exist_ok=True)
    os.makedirs(trainD_f2, exist_ok=True)
    os.makedirs(trainD_f3, exist_ok=True)
    os.makedirs(trainD_f12, exist_ok=True)
    os.makedirs(trainD_f13, exist_ok=True)
    os.makedirs(trainD_f23, exist_ok=True)
    os.makedirs(testD, exist_ok=True)
    os.makedirs(valD, exist_ok=True)

    trainFls = [
        "C18G1_2L1_1",
        "cnd1threeL1_1213061",
        "cnd1threeL1_1228061",
        "cnd1threeL1_1229062",
        "cnd1threeL1_1229063",
        "eft3RW10035L1_0125073",
        "egl5L1_0606074",
        "elt3L1_0503072",
        "elt3L1_0504073",
        "hlh1fourL1_0417071",
        "hlh1fourL1_0417076",
        "hlh1fourL1_0417077",
        "hlh1fourL1_0417078",
        "pha4A7L1_1213062",
        "pha4A7L1_1213064",
        "pha4B2L1_0125072",
        "pha4I2L_0408073",
        "unc54L1_0123071",
    ]

    valFls = [
        "cnd1threeL1_1229061",
        "pha4A7L1_1213061",
        "pha4I2L_0408071",
    ]

    trainFold1Fls = [
        "C18G1_2L1_1",
        "cnd1threeL1_1213061",
        "egl5L1_0606074",
        "hlh1fourL1_0417071",
        "hlh1fourL1_0417076",
        "pha4A7L1_1213061",
        "pha4B2L1_0125072",
    ]

    trainFold2Fls = [
        "cnd1threeL1_1229062",
        "cnd1threeL1_1229063",
        "eft3RW10035L1_0125073",
        "elt3L1_0503072",
        "hlh1fourL1_0417077",
        "pha4A7L1_1213062",
        "pha4I2L_0408071",
    ]

    trainFold3Fls = [
        "cnd1threeL1_1228061",
        "cnd1threeL1_1229061",
        "elt3L1_0504073",
        "hlh1fourL1_0417078",
        "pha4A7L1_1213064",
        "pha4I2L_0408073",
        "unc54L1_0123071",
    ]

    testFls = [
        "eft3RW10035L1_0125071",
        "eft3RW10035L1_0125072",
        "elt3L1_0503071",
        "hlh1fourL1_0417075",
        "mir61L1_1228061",
        "mir61L1_1228062",
        "mir61L1_1229062",
        "pha4I2L_0408072",
        "unc54L1_0123072",
    ]

    fmt = "." + args.out_format
    if args.out_format == "hdf":
        copy_func = shutil.copy2
    elif args.out_format == "zarr":
        copy_func = shutil.copytree

    for fl in trainFls:
        copy_func(os.path.join(args.in_dir, fl + fmt),
                  os.path.join(args.in_dir, trainD, fl + fmt))
        shutil.copy2(os.path.join(args.in_dir, fl + ".csv"), trainD)

    for fl in trainFold1Fls:
        copy_func(os.path.join(args.in_dir, fl + fmt),
                  os.path.join(args.in_dir, trainD_f1, fl + fmt))
        shutil.copy2(os.path.join(args.in_dir, fl + ".csv"), trainD_f1)

    for fl in trainFold2Fls:
        copy_func(os.path.join(args.in_dir, fl + fmt),
                  os.path.join(args.in_dir, trainD_f2, fl + fmt))
        shutil.copy2(os.path.join(args.in_dir, fl + ".csv"), trainD_f2)

    for fl in trainFold3Fls:
        copy_func(os.path.join(args.in_dir, fl + fmt),
                  os.path.join(args.in_dir, trainD_f3, fl + fmt))
        shutil.copy2(os.path.join(args.in_dir, fl + ".csv"), trainD_f3)

    for fl in trainFold1Fls:
        copy_func(os.path.join(args.in_dir, fl + fmt),
                  os.path.join(args.in_dir, trainD_f12, fl + fmt))
        shutil.copy2(os.path.join(args.in_dir, fl + ".csv"), trainD_f12)
    for fl in trainFold1Fls:
        copy_func(os.path.join(args.in_dir, fl + fmt),
                  os.path.join(args.in_dir, trainD_f13, fl + fmt))
        shutil.copy2(os.path.join(args.in_dir, fl + ".csv"), trainD_f13)

    for fl in trainFold2Fls:
        copy_func(os.path.join(args.in_dir, fl + fmt),
                  os.path.join(args.in_dir, trainD_f12, fl + fmt))
        shutil.copy2(os.path.join(args.in_dir, fl + ".csv"), trainD_f12)
    for fl in trainFold2Fls:
        copy_func(os.path.join(args.in_dir, fl + fmt),
                  os.path.join(args.in_dir, trainD_f23, fl + fmt))
        shutil.copy2(os.path.join(args.in_dir, fl + ".csv"), trainD_f23)

    for fl in trainFold3Fls:
        copy_func(os.path.join(args.in_dir, fl + fmt),
                  os.path.join(args.in_dir, trainD_f23, fl + fmt))
        shutil.copy2(os.path.join(args.in_dir, fl + ".csv"), trainD_f23)
    for fl in trainFold3Fls:
        copy_func(os.path.join(args.in_dir, fl + fmt),
                  os.path.join(args.in_dir, trainD_f13, fl + fmt))
        shutil.copy2(os.path.join(args.in_dir, fl + ".csv"), trainD_f13)

    for fl in testFls:
        copy_func(os.path.join(args.in_dir, fl + fmt),
                  os.path.join(args.in_dir, testD, fl + fmt))
        shutil.copy2(os.path.join(args.in_dir, fl + ".csv"), testD)

    for fl in valFls:
        copy_func(os.path.join(args.in_dir, fl + fmt),
                  os.path.join(args.in_dir, valD, fl + fmt))
        shutil.copy2(os.path.join(args.in_dir, fl + ".csv"), valD)

if __name__ == "__main__":
    main()
