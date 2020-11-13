from setuptools import setup, find_packages

setup(
    name='auxcpvloss',
    version='0.1',
    description='compare 3d instance segmentation methods (affinities, 3-label, signed distance transform) with and without additional auxiliary task (regress vectors to center of instance)',
    url='https://github.com/Kainmueller-Lab/aux_cpv_loss',
    author='Peter Hirsch',
    author_email='kainmuellerlab@mdc-berlin.de',
    install_requires=[
        'toml',
        'absl-py',
        'joblib',
        'pandas',
        'natsort',
        'mahotas',
        'pandas',
        'numpy',
        'h5py',
        'zarr',
        'tifffile',
        'tensorflow-gpu==2.3.1'
    ],
    packages=find_packages())
