from setuptools import find_packages, setup

setup(
    name='machine_comprehension',
    packages=['machine_comprehension'],
    version='0.1.0',
    description='Machine Comprehension models in PyTorch',
    author='@laddie132 refactored by @indonoso',
    requirements=['torch', 'numpy', 'spacy', 'h5py', 'pandas', 'PyYAML']
)