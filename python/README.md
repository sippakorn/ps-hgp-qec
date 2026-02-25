# Building the PyM4RI extension

## Install Conda, once success run following command

```conda config --set solver classic```

```conda env create -f conda_env.yml```

## Create temp dictory
```mkdir build/temp.linux-x86_64-cpython-310```

## Run the command
```python setup_m4ri.py build_ext --inplace```
when inside the ```ps-hgp-qec/python``` directory.
