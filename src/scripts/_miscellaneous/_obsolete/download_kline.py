import boto3 , re , argparse # type: ignore
import zipfile , os
import pandas as pd
from pathlib import Path

from src.proj import MACHINE

aws_info = MACHINE.local_settings('aws')
aws_access_key_id = aws_info['aws_access_key_id']
aws_secret_access_key = aws_info['aws_secret_access_key']

session = boto3.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name='cn-north-1')
s3 = session.resource('s3')
bucket = s3.Bucket('datayes-data') # type: ignore
download_path = Path('tmp_pydataloader')

def download_one_day(date , file):
    zip_file_path = download_path.joinpath(f'min.{date}.zip')
    if zip_file_path.exists(): return
    bucket.download_file(file.key , zip_file_path)

def transform_one_day(date):
    target_path = download_path.joinpath(f'min/min.{date}.feather')
    target_path.parent.mkdir(parents=True , exist_ok=True)
    if target_path.exists(): return
    zip_file_path = download_path.joinpath(f'min.{date}.zip')
    txt_file_path = f'equity_pricemin{date}.txt'

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref: 
        zip_ref.extract(txt_file_path , download_path)
    df = pd.read_csv(download_path.joinpath(txt_file_path) , sep = '\t')
    df.to_feather(target_path)

def kline_download(start = 20100104 , end = 20241226):
    os.makedirs(download_path , exist_ok=True)

    filedate = lambda x:int(re.findall(r'(\d{8})', x.key)[-1])
    filefilter = lambda x:(x.key.endswith('.zip') and os.path.basename(x.key).startswith('equity_pricemin'))
    print(start , end)
    if start <= 20230328 and end <= 20230328:
        file_list = [f for f in bucket.objects.filter(Prefix = 'equity_pricemin/') if filefilter(f)]
        file_list = [f for f in file_list if filedate(f) >= start and filedate(f) <= end]
    elif start > 20230328 and end > 20230328:
        file_list = [f for f in bucket.objects.filter(Prefix = 'snapshot/L1_services_equd_min/') if filefilter(f)]
        file_list = [f for f in file_list if filedate(f) >= start and filedate(f) <= end]
    else:
        _file_list_1 = [f for f in bucket.objects.filter(Prefix = 'equity_pricemin/') if filefilter(f)]
        _file_list_1 = [f for f in _file_list_1 if filedate(f) >= start and filedate(f) <= 20230328]
        _file_list_2 = [f for f in bucket.objects.filter(Prefix = 'snapshot/L1_services_equd_min/') if filefilter(f)]
        _file_list_2 = [f for f in _file_list_2 if filedate(f) > 20230328 and filedate(f) <= end]
        file_list = _file_list_1 + _file_list_2

    # print(f'{len(file_list)} files to download:')
    files = dict(sorted({filedate(f): f for f in file_list}.items()))
    print(files.keys())
    import concurrent.futures
    from concurrent.futures import as_completed

    def download_wrapper(args):
        date, file = args
        download_one_day(date, file)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(download_wrapper, (date , file)):date for date , file in files.items()}
        for future in as_completed(futures):
            date = futures[future]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start' , type=int , default=20100104)
    parser.add_argument('--end' , type=int , default=20241226)
    args = parser.parse_args()
    kline_download(args.start , args.end)