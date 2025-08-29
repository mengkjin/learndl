import sqlite3
from datetime import datetime

from typing import Optional, Any
from pathlib import Path

from src.proj import PATH

class DBConnHandler:
    def __init__(self, db_path: str | Path):
        self.db_path = db_path
        self.reset()

    def reset(self):
        self.check_same_thread = False
        
    @staticmethod
    def get_connection(db_path: str | Path , check_same_thread: bool = True):
        """Get database connection(using Streamlit cache)"""
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

class TaskRecorder:
    def __init__(self , type : str , name : str | None = None , key : str | None = None):
        self._type = type
        self._name = name
        self._key = key
        
        self.db_path = self.get_db_path()
        self.conn_handler = DBConnHandler(self.db_path)
        self.initialize_database()

    def get_db_path(self):
        db_path = PATH.app_db / 'task_record.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    def task_type(self , type : str | None = None):
        if type is None: return self._type
        else: return type
    
    def task_name(self , name : str | None = None):
        if name is None: 
            assert self._name is not None, 'task_name must be set'
            return self._name
        else: 
            return name
    
    def task_key(self , key : str | None = None):
        if key is None: 
            assert self._key is not None, 'task_key must be set'
            return self._key
        else: 
            return key

    def initialize_database(self):
        """Initialize database and metadata"""    
        with self.conn_handler(check_same_thread = True) as (conn, cursor):
            conn.execute('''
                CREATE TABLE IF NOT EXISTS task_meta (
                    task_type TEXT PRIMARY KEY,
                    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def get_task_types(self):
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT task_type FROM task_meta ORDER BY created_time')
            return [row['task_type'] for row in cursor.fetchall()]
        
    def ensure_task_type(self , type : str | None = None):
        """
        ensure task_type table exists, if not, create it
        """
        with self.conn_handler as (conn, cursor):
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.task_type(type)} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_name TEXT NOT NULL,
                    task_key TEXT NOT NULL,
                    success BOOLEAN DEFAULT FALSE,
                    complete_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    remark TEXT DEFAULT ''
                )
            ''')
            
            cursor.execute('''
                INSERT OR IGNORE INTO task_meta (task_type , created_time , last_updated)
                VALUES (?, ?, ?)
            ''', (self.task_type(type), datetime.now(), datetime.now()))
    
    def check_task_type(self , type : str | None = None) -> bool:
        """check if task group exists"""
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT * FROM task_meta WHERE task_type = ?', (self.task_type(type),))
            return cursor.fetchone() is not None
    
    def mark_finished(self, name: str | None = None, key: str | None = None, success: bool = True, remark: Optional[str] = None, type : str | None = None):
        """mark task finished"""
        self.ensure_task_type(type)
        with self.conn_handler as (conn, cursor):   
            cursor.execute(f'''
                INSERT INTO {self.task_type(type)} (task_name, task_key, success, complete_time, remark)
                VALUES (?, ?, ?, ?, ?)
            ''', (self.task_name(name), self.task_key(key), success * 1, datetime.now(), remark))
            
            cursor.execute('''
                UPDATE task_meta 
                SET last_updated = ?
                WHERE task_type = ?
            ''', (datetime.now(), self.task_type(type)))
    
    def is_finished(self, name: str | None = None, key: str | None = None, type : str | None = None) -> bool:
        """check if task is finished"""
        if not self.check_task_type(type): return False
        with self.conn_handler as (conn, cursor):
            cursor.execute(f'''
                SELECT sum(success) FROM {self.task_type(type)} 
                WHERE task_name = ? AND task_key = ?
            ''', (self.task_name(name), self.task_key(key)))
            return cursor.fetchone()[0] == 1
    
    def get_task_info(self, name: str | None = None, key: str | None = None, type : str | None = None):
        """get task info"""
        if not self.check_task_type(type): return None 
        with self.conn_handler as (conn, cursor):
            cursor.execute(f'''
                SELECT task_name, task_key, success, complete_time, remark 
                FROM {self.task_type(type)} 
                WHERE task_name = ? AND task_key = ?
            ''', (self.task_name(name), self.task_key(key)))
            
            result = cursor.fetchone()
            if result:
                return {
                    'task_name': result[0],
                    'task_key': result[1],
                    'complete_time': result[2],
                    'remark': result[3]
                }
            return None
    
    def clear_task(self, name: str | None = None, type : str | None = None) -> bool:
        """clear task group"""
        if not self.check_task_type(type): return False
        with self.conn_handler as (conn, cursor):
            cursor.execute(f'''
                DELETE FROM {self.task_type(type)} 
                WHERE task_name = ?
            ''', (self.task_name(name),))
            return True
    
    
    def get_finished_tasks(self, name: str | None = None, type : str | None = None):
        """get finished tasks"""
        if not self.check_task_type(type): return []
        with self.conn_handler as (conn, cursor):
            cursor.execute(f'''
                SELECT task_key, success, complete_time, remark 
                FROM {self.task_type(type)} 
                WHERE task_name = ?
            ''', (self.task_name(name),))
            return [row for row in cursor.fetchall()]

    def delete_task(self, name: str | None = None, key: str | None = None, type : str | None = None):
        """delete task"""
        if not self.check_task_type(type): return
        with self.conn_handler as (conn, cursor):
            cursor.execute(f'''
                DELETE FROM {self.task_type(type)} 
                WHERE task_name = ? AND task_key = ?
            ''', (self.task_name(name), self.task_key(key)))
    
# 使用示例
if __name__ == "__main__":
    # 创建记录器实例
    recorder = TaskRecorder('autorun')
    
    # 标记任务完成
    recorder.mark_finished("data_processing", "task_001", True, "数据处理完成")
    recorder.mark_finished("data_processing", "task_002", False, "数据清洗完成")
    
    # 检查任务是否完成
    print(f"task_001 是否完成: {recorder.is_finished('data_processing', 'task_001')}")
    print(f"task_003 是否完成: {recorder.is_finished('data_processing', 'task_003')}")
    
    # 获取任务信息
    task_info = recorder.get_task_info("data_processing", "task_001")
    print(f"任务信息: {task_info}")
    
    # 获取所有已完成任务
    finished_tasks = recorder.get_finished_tasks("data_processing")
    print(f"已完成任务: {finished_tasks}")
    
    # 获取所有任务组
    groups = recorder.get_task_types()
    print(f"所有任务组: {groups}")
    
    # 删除任务
    recorder.delete_task("data_processing", "task_002")
    
    # 清空任务组
    # recorder.clear_task_group("data_processing")