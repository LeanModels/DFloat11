from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='dfloat11',
    version='0.1.0',
    description='GPU inference for losslessly compressed (DFloat11) large language models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Tianyi Zhang',
    packages=find_packages(),
    install_requires=[
        'tqdm',
        'transformers>=4.49.0',
        'accelerate',
    ],
    extras_require={
        'cuda11': ['cupy-cuda11x'],
        'cuda12': ['cupy-cuda12x'],
    },
    python_requires='>=3.9',
)
