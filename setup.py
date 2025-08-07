from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='dfloat11',
    version='0.3.2',
    description='DFloat11: Fast and memory-efficient GPU inference for losslessly compressed LLMs and diffusion models',
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
        'transformers',
        'accelerate',
        'safetensors',
    ],
    extras_require={
        'cuda11': ['cupy-cuda11x'],
        'cuda12': ['cupy-cuda12x'],
    },
    python_requires='>=3.9',
)
