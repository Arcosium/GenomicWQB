"""순수 PP 월 한도 소진 달엔 표준·HT 컷 미달 후보를 제출하지 않는다(2026-10-20 발효)."""
import datetime as dt

from server import worker, db as _db


def _gate(monkeypatch, closed, metrics):
    w = worker.Worker.__new__(worker.Worker)
    w.user_id = 2
    worker._PP_CLOSED_CACHE.clear()
    monkeypatch.setattr(_db, 'pp_monthly_quota_hit_since', lambda uid, since: closed)
    monkeypatch.setattr(_db, 'submitted_today', lambda uid: 0)
    monkeypatch.setattr(_db, 'get_submit_mode', lambda uid: 'auto')
    return w._submit_gate(metrics, None, fail_items=[])


def test_closed_month_blocks_pp_level(monkeypatch):
    ok, reason = _gate(monkeypatch, True, {'sharpe': 1.2, 'fitness': 0.6, 'turnover': 0.1})
    assert not ok and reason.startswith('pp_month_closed')


def test_closed_month_lets_regular_cut_through(monkeypatch):
    ok, reason = _gate(monkeypatch, True, {'sharpe': 1.8, 'fitness': 1.2, 'turnover': 0.1})
    assert not reason.startswith('pp_month_closed')


def test_open_month_keeps_pp_route(monkeypatch):
    ok, reason = _gate(monkeypatch, False, {'sharpe': 1.2, 'fitness': 0.6, 'turnover': 0.1})
    assert not reason.startswith('pp_month_closed')


def test_month_start_is_new_york_first_day():
    # 2026-10-21 07:30 KST = 2026-10-20 18:30 EDT → 10/1 00:00 EDT
    ts = dt.datetime(2026, 10, 20, 22, 30, tzinfo=dt.timezone.utc).timestamp()
    assert _db.month_start_ts(ts) == dt.datetime(2026, 10, 1, 4, 0, tzinfo=dt.timezone.utc).timestamp()
    # 11/1 01:00 EDT 는 새 달
    ts2 = dt.datetime(2026, 11, 1, 5, 0, tzinfo=dt.timezone.utc).timestamp()
    assert _db.month_start_ts(ts2) == dt.datetime(2026, 11, 1, 4, 0, tzinfo=dt.timezone.utc).timestamp()
