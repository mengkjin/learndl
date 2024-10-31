import threading

def singleton_threadsafe(cls):
    instances = {}
    lock = threading.Lock()
    
    def get_instance(*args, **kwargs):
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]
    return get_instance
