import functools
import threading
from typing import Any, Callable , Iterable

class Once:
    """
    Control the execution of a function / method to only once for the same object and key

    Example1:
        Once.run(lambda *x, **y : print(x , y) , (1,2,3) , {'a':1} , 'test')
        Once.run(lambda *x, **y : print(x , y) , (1,2,3) , {'a':1} , 'test')

    Example2:
        class MyClass:
            @Once.per_object("greeting")
            def greet1(self):
                print(f"Hello1")
            
            @staticmethod
            @Once.per_object("greeting" , object = MyClass) # type: ignore  # noqa: F821
            def greet2():
                print(f"Hello2")

            @classmethod
            @Once.per_object("greeting")
            def greet3(cls):
                print(f"Hello3")

        obj = MyClass()
        obj.greet1() 
        obj.greet1() 

        obj2 = MyClass()
        obj2.greet1() 

    """
    _executed = set()
    lock = threading.Lock()

    @classmethod
    def run(cls, func: Callable, func_args : Iterable, func_kwargs : dict , mark: Any = 'default' , object: Any | None = None) -> Any:
        key = (object.__class__.__name__ , id(object) , mark)
        with cls.lock:
            execute_mark = key not in cls._executed
            if execute_mark:
                cls._executed.add(key)
        if execute_mark:
            try:
                return func(*func_args, **func_kwargs)
            except Exception:
                with cls.lock:
                    cls._executed.remove(key)
                raise
        else:
            return None

    @classmethod
    def per_object(cls , mark: Any = 'default', object: Any | None = None):
        """
        decorator for method to only execute once for the same object and key
        - if object is None , will use the first argument as the object (class for class method , instance for instance method)
        - ! if want to make a class method run only once for a instance, use Once.run(self.....) in code block instead of @Once.per_object
        - ! if want to make a static method run only once for a class, use @Once.per_object(object = MyClass) before @staticmethod
        """
        def decorator(method: Callable) -> Callable:
            @functools.wraps(method)
            def wrapper(*args, **kwargs):
                return cls.run(method , args , kwargs , mark , object if object is not None else args[0])
                # use_object = object if object is not None else args[0]
                # key = (use_object.__class__.__name__ , id(use_object) , mark)
                # if key not in cls._executed:
                #     cls._executed.add(key)
                #     return method(*args, **kwargs)
                # return None
            return wrapper
        return decorator