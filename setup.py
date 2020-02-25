from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["libblas-dev>=3", "liblapack-dev>=liblapack"]

setup(
    name="notebookc",
    version="0.0.1",
    author="Nathan Bennett",
    author_email="nbennett4122@gmail.com",
    description="A package to make predictions using Skellam regression",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/nathan-bennett/skellam",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
