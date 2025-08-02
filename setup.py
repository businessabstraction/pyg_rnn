# setup.py
from setuptools import setup, find_packages

setup(
    name="pyg_rnn",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torch_geometric"
    ],
)
