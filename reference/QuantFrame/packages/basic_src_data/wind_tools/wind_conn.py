from sqlalchemy import create_engine
from urllib.parse import quote
    
def get_wind_conn():
    """
    user need reimplement this function
    :return:
    """

    cfg = {'host': '10.0.185.137', 'port': 1521, 'user': 'wind', 'passwd': 'aaa111',
           'db': 'WindDB'}
    user, password, host, port, db = cfg["user"], cfg["passwd"], cfg["host"], cfg["port"], cfg["db"]
    password = quote(password)
    conn = create_engine(f'oracle://{user}:{password}@{db}:{port}').connect()
    return conn