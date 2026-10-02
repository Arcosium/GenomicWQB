# tests/test_submit_gate_no_quality_floor.py
# 2026-07-28 사장 지시: "그냥 제출할 수 있는건 무조건 제출하고 4개 한도 채웠으면
# 제출 대기에 넣어주는걸로." — 품질 문턱(below_value) 제거.
# 계기: 5 PASS / 0 FAIL 인 알파가 '품질 문턱 미달 0.14<0.15' 로 안 나갔다.
# 근거: 제출 실적 1·1·2·4·2 건(대개 4칸을 못 채움)인데 문턱으로 330건을 걸렀고,
#       안 쓴 예산은 이월되지 않는다.
import threading

import pytest

from server import constraint_spec, db
from server import worker as w


@pytest.fixture
def wk(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'g.db'))
    db._INITIALIZED = False
    db.init()
    uid = db.upsert_user('g@x.example', 'pw', 'GEMINI_FAKE_KEY_FOR_TEST')
    obj = w.Worker.__new__(w.Worker)
    obj.user_id = uid
    obj._stop_event = threading.Event()
    obj._lock = threading.Lock()
    obj._corr_fs_hold = set()
    monkeypatch.setattr(w.run_config, 'is_architecture_v2_enabled', lambda: True)
    yield obj
    db._INITIALIZED = False


# 문턱에 막히던 그 알파 — submission_value 가 0.15 를 못 넘는 수준
_WEAK = {'sharpe': '1.04', 'fitness': '0.29', 'turnover': '0.3444',
         'wqb_alpha_id': 'WEAK1'}


def test_weak_but_submittable_alpha_is_submitted(wk):
    ok, reason = wk._submit_gate(_WEAK, None, fail_items=[])
    assert ok is True, f'낼 수 있는데 막았다: {reason}'
    assert 'below_value' not in reason


def test_blocking_fail_is_sent_to_wqb_for_ground_truth(wk):
    """IS FAIL도 로컬에서 예측 차단하지 않고 실제 submit 응답을 받는다."""
    ok, reason = wk._submit_gate(
        {'sharpe': '1.2', 'turnover': '0.9', 'wqb_alpha_id': 'BAD1'}, None,
        fail_items=['HIGH_TURNOVER'])
    assert ok is True and reason == ''


def test_v2_quality_floor_is_observation_only_not_submit_gate(wk):
    ok, reason = wk._submit_gate(
        {'sharpe': '1.2', 'fitness': '0.5', 'wqb_alpha_id': 'WEAK2'}, None,
        fail_items=[])
    assert ok is True and reason == ''


def test_budget_exhausted_goes_to_the_waiting_queue(wk):
    for i in range(w.DAILY_SUBMIT_BUDGET):
        db.record_submit_attempt(wk.user_id, 1, i, 'rank(close)', True, 'submitted')
    ok, reason = wk._submit_gate(_WEAK, None, fail_items=[])
    assert ok is False and reason.startswith('daily_budget')
    assert reason.endswith('→queued'), reason
    assert [r['wqb_alpha_id'] for r in db.submit_queue_list(wk.user_id)] == ['WEAK1']


def test_budget_exhausted_without_alpha_id_does_not_claim_it_queued(wk):
    """넣지도 못했으면서 '→queued' 라고 적으면 라이브 피드가 거짓말한다."""
    for i in range(w.DAILY_SUBMIT_BUDGET):
        db.record_submit_attempt(wk.user_id, 1, i, 'rank(close)', True, 'submitted')
    ok, reason = wk._submit_gate({'sharpe': '1.2'}, None, fail_items=[])
    assert ok is False and reason.endswith('→미보관'), reason
    assert db.submit_queue_list(wk.user_id) == []


def test_weekly_required_check_no_longer_blocks_submission(wk):
    """주간 테마의 필수 체크는 **배수** 조건이지 제출 자격이 아니다 (2026-08-17 사장 승인).

    08-10·08-17 두 주 연속 테마가 HT 회전비율 PASS 를 요구했고, 그 사이 이 게이트가
    10 PASS / 1 FAIL 짜리 알파까지 문 앞에서 돌려보냈다. 배수 선호는 reward 의
    multiplier 항이 이미 들고 있으니 여기서 두 번 걸지 않는다.
    """
    wk._active_constraint = constraint_spec.parse(
        'region=GLB & delay=1 & universe=TOPDIV3000 & Theme Alpha test PASS')
    base = dict(_WEAK, region='GLB', _delay='1', universe='TOPDIV3000')

    for result in ('WARNING', 'PASS'):
        metrics = dict(base, _check_results={'THEME_ALPHA': result})
        ok, reason = wk._submit_gate(metrics, None, fail_items=[],
                                     code='rank(opt6_vimtaxp)')
        assert ok is True, f'{result}: {reason}'


def test_active_scope_mismatch_is_blocked_before_submit(wk):
    wk._active_constraint = constraint_spec.parse(
        'region=EUR & delay=0 & universe=TOP2500')
    metrics = dict(_WEAK, region='USA', _delay='1', universe='TOP3000')
    ok, reason = wk._submit_gate(metrics, None, fail_items=[], code='rank(opt6_vimtaxp)')
    assert ok is False and 'region=USA' in reason and 'delay=1' in reason


def test_below_wqb_min_sharpe_is_neither_submitted_nor_queued(wk):
    """샤프 1.0 미만은 PP·HT·일반 어느 부문으로도 안 나간다 — 대기열에도 넣지 않는다.

    2026-09-23 사장 지적: 예산이 차자 S=-2.52 가 budget 대기열에 들어가 있었다.
    """
    for sh in ('-2.52', '0.07', '0.99'):
        ok, reason = wk._submit_gate({'sharpe': sh, 'wqb_alpha_id': 'J' + sh}, None, fail_items=[])
        assert ok is False and reason.startswith('below_min_sharpe'), reason
    for i in range(w.DAILY_SUBMIT_BUDGET):
        db.record_submit_attempt(wk.user_id, 1, i, 'rank(close)', True, 'submitted')
    wk._submit_gate({'sharpe': '-2.52', 'wqb_alpha_id': 'JUNK'}, None, fail_items=[])
    assert db.submit_queue_list(wk.user_id) == []


def test_drain_picks_highest_sharpe_first(wk, monkeypatch):
    """13:00 에 열리는 칸은 가장 센 알파가 먼저 쓴다 — 오래된 순이 아니다."""
    for wid, sh in (('OLD_WEAK', '1.05'), ('NEW_STRONG', '2.9'), ('MID', '1.4')):
        db.submit_queue_add(wk.user_id, wqb_alpha_id=wid, kind='budget', code=wid,
                            note='', metrics={'sharpe': sh, 'wqb_alpha_id': wid})
    monkeypatch.setattr(wk, '_submitted_today', lambda: 0)
    seen = []
    monkeypatch.setattr(wk, '_submit_gate', lambda m, *a, **k: seen.append(m['wqb_alpha_id'])
                        or (False, 'test_hold'))
    wk._drain_one(1, 'u', 'p')
    assert seen == ['NEW_STRONG']


def test_rejected_queue_rows_are_purged_24h_after_rejection(wk):
    """WQB 가 거절한 대기 건은 거절 24시간 뒤 지운다 (2026-09-25 사장 지시)."""
    import time as _t
    for wid, st, note in (('REJ_OLD', 'rejected', 'rejected:LOW_SHARPE(1.1 vs 1.58) (http_403)'),
                          ('HOPELESS_OLD', 'skipped', 'rejected:PROD_CORRELATION(0.8 vs 0.7)'),
                          ('REJ_NEW', 'rejected', 'rejected:LOW_SHARPE(1.1 vs 1.58)'),
                          ('GATE_HOLD', 'skipped', 'USA/TOP1000 scope — 조건 불일치'),
                          ('QUOTA', 'pending', w.QUOTA_WAIT_NOTE)):
        db.submit_queue_add(wk.user_id, wqb_alpha_id=wid, kind='budget', metrics={'sharpe': '1.2'})
        qid = [r for r in db.submit_queue_list(wk.user_id, include_skipped=True)
               if r['wqb_alpha_id'] == wid][0]['id']
        db.submit_queue_mark(qid, st, note)
    now = _t.time()
    with db._connect() as c:     # 두 건은 25시간 전에 거절됐다
        c.execute("UPDATE submit_queue SET updated_at=? WHERE wqb_alpha_id IN "
                  "('REJ_OLD','HOPELESS_OLD','GATE_HOLD','QUOTA')", (now - 25 * 3600,))
    assert db.submit_queue_purge_rejected(wk.user_id, now=now) == 2
    left = {r['wqb_alpha_id'] for r in db.submit_queue_list(wk.user_id, include_skipped=True)}
    assert left == {'REJ_NEW', 'GATE_HOLD', 'QUOTA'}


def test_requeue_revives_unsubmitted_rows_above_wqb_floor(wk):
    for wid, st, sh, kind in (('A', 'rejected', '1.3', 'budget'), ('B', 'skipped', '2.0', 'theme'),
                              ('LOW', 'skipped', '0.4', 'budget'), ('DONE', 'submitted', '1.5', 'budget')):
        db.submit_queue_add(wk.user_id, wqb_alpha_id=wid, kind=kind, metrics={'sharpe': sh})
        qid = [r for r in db.submit_queue_list(wk.user_id, include_skipped=True)
               if r['wqb_alpha_id'] == wid][0]['id']
        db.submit_queue_mark(qid, st, 'x')
    assert db.submit_queue_requeue(wk.user_id, 1.0, note='일괄 재시도') == 2
    rows = {r['wqb_alpha_id']: r for r in db.submit_queue_list(wk.user_id, include_skipped=True)}
    assert rows['A']['status'] == rows['B']['status'] == 'pending'
    assert rows['B']['kind'] == 'budget'          # theme 은 드레인이 안 보므로 옮긴다
    assert rows['LOW']['status'] == 'skipped' and rows['DONE']['status'] == 'submitted'


def test_quota_only_rejection_is_not_a_quality_rejection():
    from server import criteria
    assert criteria.quota_only('rejected:POWER_POOL_SUBMISSION(2 vs 2) (http_403)')
    assert criteria.quota_only('rejected:REGULAR_SUBMISSION(4 vs 4) (http_403)')
    assert not criteria.quota_only('rejected:LOW_SHARPE(1.1 vs 1.58); REGULAR_SUBMISSION(4 vs 4)')
    assert not criteria.quota_only('submit_pending_timeout')
