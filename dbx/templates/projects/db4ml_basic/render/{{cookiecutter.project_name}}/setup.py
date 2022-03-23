from setuptools import find_packages, setup
from {{cookiecutter.package}} import __version__

setup(
    name="{{cookiecutter.package}}",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    version=__version__,
    description="",
    author=""
)
