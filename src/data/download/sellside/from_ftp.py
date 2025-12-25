import os , tempfile
import pandas as pd

from ftplib import FTP

from src.proj import Logger

class SellsideFTPDownloader(object):
    def __init__(self, host='47.100.224.38', user="factor", password="msjg_factor@1024"):
        self.ftp = self.ftp_login(host, user, password)
        self.ftp_param = {
            'host' : '47.100.224.38' ,
            'post' : 21 ,
            'user' : 'factor' ,
            'password' : 'msjg_factor@1024'
        }

    def ftp_login(self , host, user, password):
        ftp = FTP(host,  user, password)
        ftp.set_pasv(False)
        return ftp

    def dir(self , path = ''):
        return self.ftp.nlst(path)

    def download_file(self, remote_file, local_file):
        assert remote_file in self.ftp.nlst(os.path.dirname(remote_file)) , 'file not exists remotely'
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        with open(local_file , 'wb') as file:
            self.ftp.retrbinary('RETR ' + remote_file, file.write)

    def open_file(self, remote_file , sep = ','):
        assert remote_file in self.ftp.nlst(os.path.dirname(remote_file)) , 'file not exists remotely'
        with tempfile.TemporaryFile() as file:
            self.ftp.retrbinary('RETR ' + remote_file, file.write)
            file.seek(0)
            df = pd.read_csv(file , sep=sep)
        return df

    @classmethod
    def update(cls):
        return
        Logger.remark(f'Download: {cls.__name__} since last update!')

def main():
    connector = SellsideFTPDownloader()
    df = connector.open_file('/StockFactor_cier/cier.csv')
    Logger.stdout(df)