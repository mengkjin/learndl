import sqlite3 , shutil
from datetime import datetime
from pathlib import Path

# custom datetime adapter and converter to solve Python 3.12 deprecation warning
def adapt_datetime(dt):
    """Convert datetime to ISO format string for SQLite storage"""
    return dt.isoformat()

def convert_datetime(s):
    """Convert ISO format string back to datetime object"""
    return datetime.fromisoformat(s.decode())

# register custom adapter and converter
sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter("TIMESTAMP", convert_datetime)

class DBConnHandler:
    """
    Handler for sqlite database connection
    example:
        with DBConnHandler('path/to/db.db') as (conn, cursor):
            cursor.execute('SELECT * FROM table')
            Logger.stdout(cursor.fetchall())
    """
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.reset()

    def reset(self):
        self.check_same_thread = False
        
    @staticmethod
    def get_connection(db_path: str | Path , check_same_thread: bool = True) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(str(db_path), check_same_thread=check_same_thread)
        conn.row_factory = sqlite3.Row  # allow to access rows as dictionaries
        return conn
    
    def __call__(self , check_same_thread = False):
        self.check_same_thread = check_same_thread
        return self
        
    def __enter__(self):
        self.conn = self.get_connection(self.db_path , check_same_thread = bool(self.check_same_thread))
        self.conn.__enter__()
        self.cursor = self.conn.cursor()
        return self.conn , self.cursor
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.reset()
        self.conn.__exit__(exc_type, exc_value, exc_tb)

    def get_backup_path(self , suffix : str | None = None) -> Path:
        """Get database a backup path"""
        if suffix is None:
            suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'{self.db_path.stem}_{suffix}'
        d = self.db_path.parent / 'backup'
        d.mkdir(parents=True, exist_ok=True)
        return d / f'{backup_name}.db'
    
    def all_backup_paths(self) -> list[Path]:
        """Get all backup paths that match the database name"""
        return list(self.db_path.parent.glob(f'backup/{self.db_path.stem}_*.db'))
    
    def backup(self, suffix: str | None = None) -> Path:
        """
        Backup database and rename the original database
        :param suffix: backup suffix, if not specified, use timestamp
        :return: new database path
        """
        backup_path = self.get_backup_path(suffix)
        shutil.copy(self.db_path, backup_path)
        return backup_path
    
    def get_table_names(self) -> list[str]:
        """Get all table names in the database"""
        with self() as (conn, cursor):
            return cursor.execute('SELECT name FROM sqlite_master WHERE type = "table"').fetchall()

    def restore(self, backup_path: Path | str, delete_backup: bool = False) -> None:
        """Restore database from a backup path"""
        backup_path = Path(backup_path)
        assert backup_path.exists() , f'Backup file {backup_path} does not exist'
        assert backup_path.stem == self.db_path.stem , f'Backup file {backup_path} is not for {self.db_path}'
        with self() as (conn, cursor):
            cursor.execute('ATTACH DATABASE ? AS backup', (str(backup_path),))
            tables = self.get_table_names()
            for table in tables:
                cursor.execute(f'INSERT INTO {table} SELECT * FROM backup.{table}')
        if delete_backup:
            backup_path.unlink()