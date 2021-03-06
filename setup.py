from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["scipy>==1.2.0"]

setup(
    name="skellam",
    version="0.0.1",
    author="Nathan Bennett",
    author_email="nbennett4122@gmail.com",
    description="A package to make predictions using Skellam regression",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/nathan-bennett/skellam",
    setup_requires=['numpy'],
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
