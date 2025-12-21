#!/usr/bin/env python3
# coding: utf-8

import sys , os

if __name__ == '__main__':
    if sys.platform == 'Windows':
        os.system('runs/launch.bat')
    else:
        os.system('runs/launch.sh')