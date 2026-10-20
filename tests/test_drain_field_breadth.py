# tests/test_drain_field_breadth.py
# 드레인은 샤프보다 '이번 분기에 안 쓴 필드'와 '좋은 알파(margin·단일 데이터셋)'를 먼저 본다.
# 2026-10 실측: 샤프순 드레인이 하루 4칸을 anl69_best_pe_ratio 변주에 몰아 47건 중 15건을 차지했다.
from server import worker as w


def _row(i, code, sharpe, margin=0.0002, region='GLB'):
    return {'id': i, 'code': code,
            'metrics': {'sharpe': str(sharpe), 'margin': str(margin), 'region': region}}


def test_fresh_field_beats_higher_sharpe(monkeypatch):
    monkeypatch.setattr(w, '_datasets_used_by_code', lambda code: {'ds'})
    seen = {'anl69_best_pe_ratio'}
    old = _row(1, 'rank(anl69_best_pe_ratio)', 2.4)
    new = _row(2, 'rank(brand_new_field)', 1.2)
    assert max([old, new], key=lambda r: w._drain_key(r, seen)) is new


def test_good_margin_single_dataset_breaks_tie(monkeypatch):
    monkeypatch.setattr(w, '_datasets_used_by_code', lambda code: {'ds'})
    thin = _row(1, 'rank(f1)', 2.0, margin=0.0003)
    good = _row(2, 'rank(f2)', 1.5, margin=0.0011)
    assert max([thin, good], key=lambda r: w._drain_key(r, set())) is good


def test_asi_needs_15bps(monkeypatch):
    monkeypatch.setattr(w, '_datasets_used_by_code', lambda code: {'ds'})
    r = _row(1, 'rank(f1)', 1.5, margin=0.0012, region='ASI')
    assert w._drain_key(r, set())[1] is False
    r = _row(1, 'rank(f1)', 1.5, margin=0.0012, region='GLB')
    assert w._drain_key(r, set())[1] is True


def test_two_datasets_is_not_good(monkeypatch):
    monkeypatch.setattr(w, '_datasets_used_by_code', lambda code: {'a', 'b'})
    r = _row(1, 'rank(f1)', 1.5, margin=0.002)
    assert w._drain_key(r, set())[1] is False


def test_same_fields_fall_back_to_sharpe(monkeypatch):
    monkeypatch.setattr(w, '_datasets_used_by_code', lambda code: {'ds'})
    seen = {'f1'}
    lo, hi = _row(1, 'rank(f1)', 1.3), _row(2, 'ts_rank(f1, 20)', 1.9)
    assert max([lo, hi], key=lambda r: w._drain_key(r, seen)) is hi
