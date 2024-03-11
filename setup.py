from setuptools import setup, find_packages

setup(
    name="dynamic_yolo_mosaic_generator",
    version="0.1",
    description="Advanced toolkit for computer vision tasks with Python.",
    author="Anton Shalin",
    author_email="anton.shalin@gmail.com",
    packages=find_packages(),
    install_requires=[line.strip() for line in open("requirements.txt", "r")],
)