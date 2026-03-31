from setuptools import setup, find_packages

setup(
    name="amazon-review-analyzer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.9",
)
