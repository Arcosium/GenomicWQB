# tests/test_scan_dataset.py — 스캔 요약이 과제 2 양식 숫자(저커버리지·상수·중복·유망)를 바로 세는지.
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    'scan_dataset', os.path.join(os.path.dirname(__file__), '..', 'scripts', 'scan_dataset.py'))
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)


def test_summary_counts():
    rows = [
        {'field': 'a', 'coverage': 0.9, 'sharpe': 2.1, 'turnover': 0.1, 'fitness': 1.0},
        {'field': 'a_2', 'coverage': 0.9, 'sharpe': 2.1, 'turnover': 0.1, 'fitness': 1.0},
        {'field': 'b', 'coverage': 0.2, 'sharpe': -1.7, 'turnover': 0.2, 'fitness': -0.5},
        {'field': 'c', 'coverage': 0.9, 'sharpe': 0.0, 'turnover': 0.0, 'fitness': 0.0},
        {'field': 'd', 'coverage': 0.9, 'sharpe': None},
    ]
    s = scan.summarize(rows)
    assert (s['fields'], s['measured'], s['low_coverage'], s['constant'], s['duplicate']) == (5, 4, 1, 1, 1)
    assert [p['field'] for p in s['promising']] == ['a', 'b']
