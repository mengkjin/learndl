#!/usr/bin/env python3
# coding: utf-8

import os , platform

if __name__ == '__main__':
    if platform.system() == 'Windows':
        os.system('cmd /c runs\\launch.bat')
    else:
        os.system('runs/launch.sh')