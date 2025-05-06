from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='dfloat11',
    version='0.2.0',
    description='GPU inference for losslessly compressed (DFloat11) Large Language Models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Tianyi Zhang',
    packages=find_packages(),
    package_data={
        "dfloat11": ['decode.ptx'],
    },
    include_package_data=True,
    install_requires=[
        'tqdm',
        'transformers>=4.51.0',
        'accelerate',
        'safetensors',
    ],
    extras_require={
        'cuda11': ['cupy-cuda11x'],
        'cuda12': ['cupy-cuda12x'],
    },
    python_requires='>=3.9',
)
