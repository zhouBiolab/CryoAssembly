
import multiprocessing


def worker_count(workers):
    return max(1, int(workers or 1))


def pool_context(start_method=None):
    if start_method:
        return multiprocessing.get_context(start_method)
    return multiprocessing


def open_pool(processes, start_method=None):
    return pool_context(start_method).Pool(processes=processes)


def close_pool(pool):
    pool.close()
    pool.join()
