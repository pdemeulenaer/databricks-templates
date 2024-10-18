from setuptools import setup, find_packages, find_namespace_packages
# import codecs
import sys
sys.path.append("./src")
import datetime
import {{cookiecutter.module_slug}}


setup(
    name='{{cookiecutter.module_slug}}',
    version={{cookiecutter.module_slug}}.__version__ + "+" + datetime.datetime.utcnow().strftime("%Y%m%d.%H%M%S"),
    url="https://databricks.com",
    author="Northell",
    author_email="philippe.demeulenaer@northell.com",
    description="wheel file based on {{cookiecutter.project_slug}}/src",
    packages=find_packages(where="./src"),
    package_dir={"": "src"},
    entry_points={
        "packages": [
            "data_extraction={{cookiecutter.module_slug}}.data_pipeline.task1:main",
            "data_visualization={{cookiecutter.module_slug}}.data_pipeline.task2:main",
            "training={{cookiecutter.module_slug}}.model.train:main",
            "evaluation={{cookiecutter.module_slug}}.model.evaluate:main",
        ]
    },
    install_requires=[
        # Dependencies in case the output wheel file is used as a library dependency.
        # For defining dependencies, when this package is used in Databricks, see:
        # https://docs.databricks.com/dev-tools/bundles/library-dependencies.html
        "setuptools"
    ],
    # package_data={'global_toolkit': ['**/*.yaml']},
    package_data={"": ["*.yaml", "*.csv", "*.pkl"]},  # Include all YAML, CSV and PKL files in the package
    include_package_data=True,  # Include package data specified by package_data
)
