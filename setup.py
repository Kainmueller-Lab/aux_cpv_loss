from setuptools import setup, find_packages

setup(
    name='auxcpvloss',
    version='1.0',
    install_requires=[
        'toml',
        'joblib',
        'pandas',
        'natsort',
        'mahotas',
        'zarr',
        'tifffile',
    ],
    packages=find_packages())
