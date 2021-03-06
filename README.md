# MPRI

## Overview

This project is a fast algorithm to calculate MPRI, which is inspired by this paper[1]

## Requirements

It is requires Python3 and is compatible both in Windows and Linux.

### Install

1. The most recommanded way is using `pipenv`, which is a modern virtualenv and package management tool.

    `pip install pipenv`

    After pipenv installed, you can use that to install required packages and a pure virtual enviroment.

    `pipenv install`

    Then entry this new virtual enviroment.

    `pipenv shell`

    Note that if you intall in this way, you need to entry the virtual enviroment by `pipenv shell` + `python ***` or `pipenv run python ***` every time you run it.

2. If you don't want to learn to use `pipenv`, you can also install packages in the traditional way.

    `pip install -r requirements.txt`

## Usage

```
python main.py MODE --image/-i IMAGE
                    --lable/-l LABLE
                    [--output/-o OUTPUT]
                    [--data/d DATA]

Mode:
    calc:                   Calculate and output the pons area, midbrain area, MCP
                            width, SCP width and MPRI.
    seg:                    Segment MCP, SCP and save to output image but not
                            calculate MPRI, etc. Note that --output OUTPUT must be
                            given in this mode.
    both:                   Equal to runing both calc mode and seg mode.

Required arguments:
    --image/-i IMAGE:       The image path IMAGE to be segmented or calculated.
    --lable/-l LABLE:       The lable path LABLE of correspounding image
                            from AccuBrain.

Optional arguments:
    --output/-o OUTPUT:     The output path OUTPUT to save segmentation results.
                            It must be given if the mode is seg or both and does
                            not work if the mode is calc.
    --data/-d DATA:         The output path DATA to save calculation results. If
                            given, calculation results would be saved to DATA,
                            otherwise shown in the screen.
```

## Reference

[1] Nigro S, Arabia G, Antonini A, Weis L, Marcante A, Tessitore A et al. Magnetic Resonance Parkinsonism Index: diagnostic accuracy of a fully automated algorithm in comparison with the manual measurement in a large Italian multicentre study in patients with progressive supranuclear palsy. European Radiology. 2016 Oct 19;1-11. https://doi.org/10.1007/s00330-016-4622-x