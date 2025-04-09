from typing import Any, List, Tuple, Union


class DummyClass:
    def __getattr__(self, name):
        # 返回另一个 DummyClass 实例，从而支持链式调用
        return (
            DummyClass()
        )  # lambda *args, **kwargs: None  # 返回一个直接 pass 的空函数

    def __call__(self, *args, **kwargs):
        # 允许 DummyClass 实例被调用（如 canvas.ax.scatter()），直接 pass
        return None
