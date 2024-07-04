from setuptools import setup, find_packages

setup(
    name='genaibench',
    version='0.0.1',
    description='Official Codes for GenAI-Bench, as part of GenAI-Arena',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Dongfu Jiang',
    author_email='dongfu.jiang@uwaterloo.ca',
    packages=find_packages(),
    url='https://github.com/TIGER-AI-Lab/GenAI-Bench',
    install_requires=[
        "transformers",
        "sentencepiece",
        "torch",
        "Pillow",
        "torch",
        "accelerate",
        "torchvision",
        "datasets",
        "tqdm",
        "numpy",
        "prettytable",
        "fire",
        "datasets",
        "openai",
        "tiktoken",
        "av",
        "decord",
        "mantis-vl",
        "protobuf",
        "opencv-python",
        "peft"
    ],
    extras_require={}
)



# change it to pyproject.toml
# [build-system]
