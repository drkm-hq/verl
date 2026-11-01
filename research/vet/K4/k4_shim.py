"""CPU stand-ins for the two verl imports bucketed_weight_transfer.py makes (see k4_cast_order_cpu.py)."""
import torch


def get_device_name():
    return "cpu"


def get_device_id():
    return 0


def is_support_ipc():
    return False


class _Dev:
    @staticmethod
    def synchronize():
        pass

    @staticmethod
    def empty_cache():
        pass

    @staticmethod
    def ipc_collect():
        pass


def get_torch_device():
    return _Dev


async def ensure_async_iterator(it):  # verbatim semantics of verl.workers.rollout.utils.ensure_async_iterator
    if hasattr(it, "__aiter__"):
        async for x in it:
            yield x
    else:
        for x in it:
            yield x
