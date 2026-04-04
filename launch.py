#!/usr/bin/env python3
# coding: utf-8

import platform , subprocess

if __name__ == '__main__':
    if platform.system() == 'Windows':
        subprocess.run('cmd /c runs\\launch.bat')
    else:
        subprocess.run('runs/launch.sh')