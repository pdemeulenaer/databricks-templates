# Introduction 
This project aims to create a repository of templates used to kick off new Databricks-based projects  

# Getting Started
## Install Cookie Cutter
To make use of scripted project creation, first install Cookie Cutter globally
```bash
pip install --user cookiecutter
```

## Run template
In order to create a project based on existing template, first navigate to directory where the project needs to be created
and then run Cookie Cutter with an argument pointing to desired template.
For example, if databricks-templates was cloned into the current directory and user is aiming to create llm fine-tuning project in
the same current directory:
```bash
cookiecutter databricks-templates/python-package-dbx-llms
```

More information can be found here: https://cookiecutter.readthedocs.io/en/latest/usage.html#grab-a-cookiecutter-template

# Contributing
Each new template should be a separate directory added to main directory of this repository.
To create templates, please follow this tutorial: https://cookiecutter.readthedocs.io/en/stable/tutorials/tutorial2.html#tutorial2

# Template maintenance

