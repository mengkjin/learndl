"""
Sell-side FTP downloader for file-transfer-based data delivery.

Note: ``SellsideFTPDownloader.update()`` is currently a no-op (dead code after
an early ``return`` statement).  The FTP infrastructure is in place but the
actual update logic is not yet implemented.  See TODO_data.md item C1.
"""
import os , tempfile
import pandas as pd

from ftplib import FTP
from typing import Any

from src.proj import Logger , MACHINE

class SellsideFTPDownloader(object):
    """
    FTP client for downloading sell-side data delivered via file transfer.

    Connects to the FTP server using credentials from ``MACHINE.secret``.
    Provides ``dir()``, ``download_file()``, and ``open_file()`` for listing
    and downloading remote CSV files.
    """
    def __init__(self, source='msjg'):
        assert source in MACHINE.secret.get('accounts' , 'sellside') , f'{source} is not a valid source name, check .secret/accounts.yaml[sellside]'
        self.ftp_param : dict[str , Any] = MACHINE.secret.get('accounts' , f'sellside/{source}')
        type = self.ftp_param.pop('type')
        assert type.startswith('ftp') , f'{source} is not a valid ftp source : {self.ftp_param}'
        if type.endswith('.disabled'):
            Logger.alert1(f'{source} is disabled')
        self.ftp = self.ftp_login(**self.ftp_param)

    def ftp_login(self , host, user, password , **kwargs):
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
        Logger.note(f'Download: {cls.__name__} since last update!')

    @classmethod
    def available_sources(cls) -> list[str]:
        return [key for key , value in MACHINE.secret.get('accounts' , 'sellside').items() if value['type'] == 'ftp']

def main():
    connector = SellsideFTPDownloader()
    df = connector.open_file('/StockFactor_cier/cier.csv')
    Logger.stdout(df)