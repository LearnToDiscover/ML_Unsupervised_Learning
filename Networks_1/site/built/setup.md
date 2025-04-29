---
title: Setup
---

## Setting up a virtual environment:

In Python, the use of virtual environments allows you to avoid installing Python packages globally which could break system tools or other projects.  Each virtual environment has its own Python binary (which matches the version of the binary that was used to create this environment) and can have its own independent set of installed Python packages in its site directories.

A virtual environment can be created by executing the command `venv` in your Terminal (Mac OS and Unix) or at the command line prompt (Windows):

```
python3 -m venv pyML
```

By running this command, a new environment will be installed in your home directory.

The environment can be activated with the following command:

```
source pyML/bin/activate 
```

Next, the Python packages required for this lesson can be installed:

```
pip3 install pandas numpy matplotlib scipy networkx
```

This environment kernel must then be to be added to your Jupyter Notebook. This can be done as:

```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=firstEnv
```

After running these commands, you will be able to select your virtual environment from the `Kernel` tab in your IDE of choice.

## Data:

The data files used in this lesson:

- [BreastCancerNetwork.csv](data/BreastCancerNetwork.csv)
- [Network2.csv](data/Network2.csv)