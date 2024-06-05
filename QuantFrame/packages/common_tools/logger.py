import os
import time


def write_log(file_name_: str, line_: str):
    f = open(file_name_, "a")
    f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + line_ + "\n")
    f.close()
