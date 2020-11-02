"""
Basecaller parallel data loader
"""
import re
import uuid
import threading
import multiprocessing
import importlib
from multiprocessing import pool
from collections import deque
from typing import Dict, Tuple, List, Sized, Iterable, Callable, Union, Optional, Any


__all__ = [
    'CountDownLatch', 'BatchEnumerator', 'ParallelLoader',
    'set_function', 'register_function', 'register_function_provider',
    'pop_task_result', 'submit_parallel_task', 'create_parallel_loader'
]


class CountDownLatch:
    def __init__(self, count=1):
        self.count = count
        self.lock = threading.Condition()

    def count_down(self):
        self.lock.acquire()
        self.count -= 1
        if self.count <= 0:
            self.lock.notify_all()
        self.lock.release()

    def wait(self):
        self.lock.acquire()
        while self.count > 0:
            self.lock.wait()
        self.lock.release()


class BatchEnumerator:
    """
    NOT Thread-Safe.
    """
    def __init__(self, source: Iterable[Any]):
        self.__source = source
        self.__iter = iter(source)
        self.__fallback = []

    def has_next(self):
        return len(self.__fallback) > 0 or self.__iter_has_next()

    def next(self):
        if len(self.__fallback) > 0:
            return self.__fallback.pop(0)
        return next(self.__iter)

    def fallback(self, data):
        self.__fallback.insert(data, 0)

    def __iter_has_next(self) -> bool:
        try:
            next_val = next(self.__iter)
            self.__fallback.append(next_val)
            return True
        except StopIteration:
            return False


class ParallelLoader:
    def __init__(self, p: pool.Pool, data: Union[BatchEnumerator, Iterable], func: Union[str, Callable] = '',
                 max_preload_count: int = 0, close_pool: bool = False):
        self.__pool = p
        self.__batch_enumerator = data if isinstance(data, BatchEnumerator) else BatchEnumerator(data)
        self.__func_name = ParallelLoader.parse_func(func)
        self.__max_preload_count = max_preload_count
        self.__close_pool = close_pool
        self.__thread_lock = threading.Lock()
        self.__thread: Optional[threading.Thread] = None
        self.__data_lock = threading.Lock()
        self.__task_mutex = threading.Lock()
        self.__cond_not_empty = threading.Condition(self.__task_mutex)
        self.__cond_not_full = threading.Condition(self.__task_mutex)
        self.__task_queue = deque()

    @staticmethod
    def parse_func(func: Union[str, Callable]):
        if isinstance(func, str):
            return func
        if func.__name__ == '<lambda>':
            raise TypeError('lambda not supported')
        # noinspection PyUnresolvedReferences
        return f'{func.__name__}@{str(func.__module__)}'

    def start(self, clear=False):
        with self.__thread_lock:
            if not self.__thread or not self.__thread.is_alive():
                if clear:
                    with self.__data_lock:
                        self.__task_queue.clear()
                self.__thread = threading.Thread(target=self.__thread_run, daemon=True)
                self.__thread.start()

    def stop(self):
        with self.__thread_lock:
            if self.__thread:
                self.__thread = None

    def has_next(self):
        with self.__task_mutex, self.__data_lock:
            return len(self.__task_queue) or self.__batch_enumerator.has_next()

    def next(self):
        if not self.has_next():
            raise StopIteration('end reached')
        with self.__cond_not_empty:
            while len(self.__task_queue) <= 0:
                self.__cond_not_empty.wait(0.5)
            task_uid = self.__task_queue.popleft()
            self.__cond_not_full.notify()
        return pop_task_result(task_uid)

    def __thread_run(self):
        current_thread = threading.current_thread()
        while self.__check_thread(current_thread):
            with self.__data_lock:
                if not self.__batch_enumerator.has_next():
                    self.__release()
                    break
            with self.__cond_not_full:
                with self.__data_lock:
                    batch_data = self.__batch_enumerator.next()
                if not isinstance(batch_data, Iterable):
                    batch_data = [batch_data]
                while 0 < self.__max_preload_count <= len(self.__task_queue):
                    if not self.__check_thread(current_thread):
                        with self.__data_lock:
                            self.__batch_enumerator.fallback(batch_data)
                        break
                    self.__cond_not_full.wait(0.5)
                task_uid = submit_parallel_task(self.__pool, self.__func_name, batch_data)
                if not task_uid:
                    break
                self.__task_queue.append(task_uid)
                self.__cond_not_empty.notify_all()

    def __check_thread(self, cur_thread: threading.Thread = None):
        if cur_thread is None:
            cur_thread = threading.current_thread()
        with self.__thread_lock:
            return self.__thread is not None and self.__thread == cur_thread

    def __release(self):
        if self.__close_pool:
            self.__pool.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.__release()


__RegexPattern = re.compile('').__class__
__RegexMatch = re.match('', '').__class__

# Function registries
__named_functions: Dict[str, Callable[[Any], Any]] = dict()
__function_providers: List[Tuple[__RegexPattern, Callable[[__RegexMatch], Callable[[Any], Any]]]] = list()
__temp_functions: Dict[str, Callable[[Any], Any]] = dict()

# Task manager
__task_results_lock = threading.Lock()
__task_results: Dict[uuid.UUID, Tuple[CountDownLatch, List[Any]]] = dict()


def __process_data(index: Tuple[uuid.UUID, int], func_name: str, data: Any):
    if '@' in func_name:
        fn_name, fn_module_name = func_name.split('@', 1)
        fn_module = importlib.import_module(fn_module_name)
        if not hasattr(fn_module, fn_name):
            raise NameError(f'function \'{func_name}\' not found, module: {fn_module_name}')
        process_fn = getattr(fn_module, fn_name)
    else:
        if func_name in __named_functions:
            process_fn = __named_functions[func_name]
        elif func_name in __temp_functions:
            process_fn = __temp_functions[func_name]
        else:
            process_fn = None
            for pattern, provider in __function_providers:
                match = pattern.fullmatch(func_name)
                if not match:
                    continue
                process_fn = provider(match)
                if process_fn is None:
                    continue
                __temp_functions[func_name] = process_fn
                break
            if process_fn is None:
                raise NameError(f'function \'{func_name}\' is not registered')
    return index, process_fn(data)


def __collect_result(callback_data: Tuple[Tuple[uuid.UUID, int], Any]):
    ((batch_index, item_idx), result) = callback_data
    (count_down_latch, result_list) = __task_results[batch_index]
    result_list[item_idx] = result
    count_down_latch.count_down()


def __error_callback(ex: BaseException):
    raise ex


def set_function(process_fn: Callable[[Any], Any]):
    register_function('', process_fn)


def register_function(name: str, process_fn: Callable[[Any], Any]):
    if '@' in name:
        raise ValueError(f'malformed name with \'@\': {name}')
    __named_functions[name] = process_fn


def register_function_provider(pattern: Union[str, __RegexPattern],
                               function_provider: Callable[[__RegexMatch], Callable[[Any], Any]] = None):
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    if not isinstance(pattern, __RegexPattern):
        raise TypeError(pattern)
    if function_provider is not None:
        __function_providers.append((pattern, function_provider))
        return

    # decorator support
    def _decorator(func):
        __function_providers.append((pattern, func))
        return func
    return _decorator


def pop_task_result(uid: uuid.UUID) -> Any:
    if uid not in __task_results:
        raise KeyError(f'task not found: {uid}')
    with __task_results_lock:
        __task_results[uid][0].wait()
    with __task_results_lock:
        results = __task_results[uid][1]
        del __task_results[uid]
    return results


def submit_parallel_task(p: pool.Pool, func_name: str, sub_items: Iterable[Any]) -> Optional[uuid.UUID]:
    if not isinstance(sub_items, Sized):
        sub_items = list(sub_items)
    item_count = len(sub_items)
    count_down_latch = CountDownLatch(item_count)
    result_list = [None] * item_count
    while True:
        uid = uuid.uuid4()
        with __task_results_lock:
            if uuid not in __task_results:
                __task_results[uid] = (count_down_latch, result_list)
                break
    for idx, item in enumerate(sub_items):
        p.apply_async(__process_data, args=[(uid, idx), func_name, item],
                      callback=__collect_result, error_callback=__error_callback)
    return uid


def create_parallel_loader(parallel_level: int, data: Union[BatchEnumerator, Iterable],
                           max_preload_count: int = 0, func: Union[str, Callable] = '') -> ParallelLoader:
    """create parallel data loader

    Args:
        parallel_level: num_process
        data: iterator whose element is iterator.
        max_preload_count: max preload count.
        func: process function to the elements of elements in `data`.

    Returns: parallel data loader.

    """
    return ParallelLoader(multiprocessing.Pool(parallel_level), data, func, max_preload_count, close_pool=True)
