"""Builders for tests/test_turnkey_units.py (OF-0011 add 90 revision 1): units that only synchronise."""
import os


def counter(sync, n, unit):
    answers = []
    for k in range(n):
        answers.append(sync(k, dict(unit=unit, k=k, pid=os.getpid())))
    return dict(answers=answers, pid=os.getpid())


def failing(sync, n):
    for k in range(n):
        sync(k, dict(unit="f", k=k))
    raise ValueError("planned failure")


def skipping(sync):
    sync(0)
    sync(2)
