from setuptools import setup, find_packages

with open("requirements-mlops.txt") as f:
    requirements = f.read().splitlines()

setup(
    name = "COLORECTAL CANCER PATIENT SURVIVAL PREDICTION",
    version = "0.1",
    author = "Rhitwik Prakash",
    packages = find_packages(),
    install_requires = requirements
)
