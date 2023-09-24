from setuptools import setup, find_packages

setup(
    name='gTDR',
    version='1.0',
    packages=[
        "gTDR",
        *["gTDR." + p for p in find_packages(where="./gTDR")],
    ],
    package_dir={"": ".",},
    install_requires=[
        'scipy>=0.19.0',
        'numpy>=1.12.1',
        'pandas>=0.19.2',
        'pyyaml',
        'statsmodels',
        'wrapt',
        'tensorflow>=1.3.0',
        'tables',
        'future',
        'scikit-learn',
        'chardet',
        'ogb',
        'matplotlib',
        'seaborn',
        'gdown',
    ],
    author="MIT-IBM Watson AI Lab, IBM Research",
    author_email="xieyi@mit.edu, chenjie@us.ibm.com",
    keywords=[
        "graph neural network",
        "graph data",
        "temporal data",
        "dynamic data",
    ],
)

