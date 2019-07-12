import argparse
import os
import sys
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest="in_dir", required=True)
    parser.add_argument('-o', dest="out_dir", required=True)
    args = parser.parse_args()

    trainD = os.path.join(args.out_dir, "train")
    testD = os.path.join(args.out_dir, "test")
    valD = os.path.join(args.out_dir, "val")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(trainD, exist_ok=True)
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

    for fl in trainFls:
        shutil.copy2(os.path.join(args.in_dir, fl+".hdf"), trainD)
        shutil.copy2(os.path.join(args.in_dir, fl+".csv"), trainD)

    for fl in testFls:
        shutil.copy2(os.path.join(args.in_dir, fl+".hdf"), testD)
        shutil.copy2(os.path.join(args.in_dir, fl+".csv"), testD)

    for fl in valFls:
        shutil.copy2(os.path.join(args.in_dir, fl+".hdf"), valD)
        shutil.copy2(os.path.join(args.in_dir, fl+".csv"), valD)

if __name__ == "__main__":
    main()
