from .basic import TushareFetecher
from .download import TushareDownloader
from .model import TushareModelCalculator

def main_task():
    TushareDownloader.proceed()
    TushareModelCalculator.proceed()