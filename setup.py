from setuptools import setup, find_packages

setup(
    name="pyg-rnnconv",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="RNNConv module for PyTorch Geometric",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/businessabstraction/pyg_rnn",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torch-geometric",
        "torch-scatter",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
