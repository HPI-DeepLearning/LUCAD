### Prototype for lung nodule detection

This repository contains multiple useful scripts to train (and score) models
with *mxnet* for the [LUNA challenge](https://luna16.grand-challenge.org/).

## Dependencies

First, [add this library to your python path](#add-to-python-path).
Then install the following libraries if they are not already installed:

1. - A few small [libraries](#libraries)
2. - [mxnet](#mxnet)
3. - [OpenCV](#opencv)

### Libraries

Install these with pip:
```bash
pip install -r requirements.txt
```

### mxnet

Either [install manually](https://mxnet.incubator.apache.org/get_started/install.html)
(needed to enable GPU support) or use pip:
```bash
pip install mxnet
```

### OpenCV

Either [install manually](http://opencv.org/releases.html) or use pip:
```bash
pip install opencv-python
```

### Add to python path

Add the `scripts` path to your `PYTHONPATH`.

If you use `virtualenv` you can use a .pth file, e.g. execute this command
from the git root repository (replace `[YOUR_VENV_PATH]`):
```bash
echo "$(pwd)/scripts" > [YOUR_VENV_PATH]/lib/python2.7/site-packages/lucad.pth
```

Otherwise add something like this to your `.bashrc` or `.zshrc` and
logout/login or reboot:

```bash
export PYTHONPATH="$PYTHONPATH:[GIT_ROOT]/scripts"
```

## Execute scripts

Please execute all scripts from this directory, i.e. to start the viewer use this
command from git root directory:
```
python scripts/viewer/viewer.py path/to/LUNA/data
```

