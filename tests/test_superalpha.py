# tests/test_superalpha.py
# ⑤ 슈퍼알파: selection/combo 표현식 생성(순수부) + superalpha_runs db 헬퍼.
import random

import pytest

from server import db, superalpha


def test_selection_expression_per_category():
    s = superalpha.selection_expression('model')
    assert 'in(datacategories, "model")' in s
    assert '(2 - self_correlation)' in s          # 기존 알파와 덜 겹치는 순


def test_build_candidates_rotate_categories_on_glb():
    n = len(superalpha.CATEGORIES)
    a = superalpha.build_candidates(0, n=3, rng=random.Random(1))
    b = superalpha.build_candidates(1, n=3)
    assert [c['category'] for c in a] == list(superalpha.CATEGORIES[:3])
    assert [c['category'] for c in b] == list(superalpha.CATEGORIES[1:4])   # 날마다 시작점이 돈다
    assert len({c['selection'] for c in a}) == 3                           # 후보마다 다른 재료
    assert len(superalpha.build_candidates(n - 1, n=n)) == n               # 끝에서 앞으로 감김
    st = a[0]['settings']
    assert (st['region'], st['universe'], st['neutralization']) == ('GLB', 'MINVOL1M', 'COUNTRY')
    assert st['selectionLimit'] == superalpha.SELECTION_LIMIT >= 10        # WQB 최소 10
    assert st['selectionHandling'] == superalpha.SELECTION_HANDLING


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    tmp_db = str(tmp_path / 'sa.db')
    monkeypatch.setattr(db, 'DB_PATH', tmp_db)
    db._INITIALIZED = False
    db.init()
    uid = db.upsert_user('u', 'p', 'GEMINI_FAKE_KEY_FOR_TEST')
    yield uid
    db._INITIALIZED = False


def test_superalpha_run_lifecycle(isolated_db):
    uid = isolated_db
    rid = db.superalpha_start(uid, 37, 'd = 37; ...')
    runs = db.superalpha_runs_list(uid)
    assert runs[0]['id'] == rid and runs[0]['status'] == 'running'
    db.superalpha_finish(rid, 'done', [{'combo': 'rank(stats.pnl)',
                                        'alpha_id': 'abc'}])
    runs = db.superalpha_runs_list(uid)
    assert runs[0]['status'] == 'done'
    assert runs[0]['results'][0]['alpha_id'] == 'abc'


# ── 권한 미보유 처리 (2026-07-27 실측) ──────────────────────────────────────
# CONSULTANT 권한만 있는 계정에서 실제로 받은 응답:
#   400 {"type":["Not permissioned for super simulations"]}
# super simulation 은 CONSULTANT 와 **별개 권한**이다.

class _NotPermissionedClient:
    """submit 마다 권한 거절을 돌려주는 가짜 클라이언트."""

    def __init__(self):
        self.calls = 0

    def authenticate(self):
        return True

    def submit_super_simulation(self, selection, combo, settings):
        self.calls += 1
        return 'NOT_PERMISSIONED'


def test_run_stops_immediately_when_account_lacks_permission(isolated_db, monkeypatch):
    """남은 후보를 더 돌려도 전부 같은 400 이다 — 첫 건에서 끝내고 사유를 남긴다."""
    uid = isolated_db
    fake = _NotPermissionedClient()
    import server.wqb_api as wqb_api
    monkeypatch.setattr(wqb_api, 'WqbApiClient', lambda u, p: fake)

    rid = superalpha.run(uid, 'u', 'p', seed_plus=10, n=6)
    assert fake.calls == 1, '권한 거절인데 후보 6개를 다 시도했다'
    row = db.superalpha_runs_list(uid)[0]
    assert row['id'] == rid and row['status'] == 'error'
    assert 'super simulation 권한' in (row['error'] or '')


# ── FAIL 0 이면 제출 (2026-10-31 권한 확인 뒤) ───────────────────────────────

class _FakeClient:
    """첫 시뮬 요청은 429, 이후 결과는 FAIL → FAIL 0 → FAIL 0 순서로 돌려준다."""

    def __init__(self):
        self.sim_calls = 0
        self.submitted = []
        self.n = 0

    def authenticate(self):
        return True

    def submit_super_simulation(self, selection, combo, settings):
        self.sim_calls += 1
        return 'RATE_LIMITED' if self.sim_calls == 1 else 'loc'

    def poll(self, loc, stop_event=None):
        self.n += 1
        return {'alpha': f'A{self.n}', 'status': 'COMPLETE'}

    def harvest_alpha(self, aid):
        fail = [{'name': 'HIGH_TURNOVER'}] if aid == 'A1' else []
        return {'metrics': {'sharpe': 2.7}, 'is_status': {'pass': ['LOW_SHARPE'], 'fail': fail}}

    def set_super_description(self, aid, selection, combo):
        assert len(selection) >= 100 and len(combo) >= 100, 'WQB 는 100자 미만 설명을 거절한다'
        self.described = aid

    def submit_alpha(self, aid, stop_event=None, deadline_s=None):
        assert getattr(self, 'described', None) == aid, '설명 없이 제출했다'
        self.submitted.append(aid)
        return True, 'submitted'


def test_run_submits_first_clean_result_once(isolated_db, monkeypatch):
    uid = isolated_db
    fake = _FakeClient()
    import server.wqb_api as wqb_api
    monkeypatch.setattr(wqb_api, 'WqbApiClient', lambda u, p: fake)
    monkeypatch.setattr(superalpha.time, 'sleep', lambda s: None)

    superalpha.run(uid, 'u', 'p', seed_plus=10, n=3)
    assert fake.sim_calls == 4, '429 를 받은 후보를 버렸다'
    assert fake.submitted == ['A2'], 'FAIL 있는 결과를 냈거나, 성공 뒤에도 또 냈다'
    res = db.superalpha_runs_list(uid)[0]['results']
    assert res[1]['submit'] == {'ok': True, 'status': 'submitted'}
    assert 'submit' not in res[0] and 'submit' not in res[2]
    # 일반 제출 하루 4건 예산(submit_attempts 집계)을 먹지 않는다.
    assert db.submitted_today(uid) == 0


def test_descriptions_meet_wqb_minimum():
    assert all(len(superalpha.selection_description(c)) >= 100 for c in superalpha.CATEGORIES)
    assert len(superalpha.COMBO_DESCRIPTION) >= 100


def test_run_does_not_submit_in_list_mode_or_without_checks(isolated_db, monkeypatch):
    uid = isolated_db
    import server.wqb_api as wqb_api
    monkeypatch.setattr(superalpha.time, 'sleep', lambda s: None)

    fake = _FakeClient()
    monkeypatch.setattr(wqb_api, 'WqbApiClient', lambda u, p: fake)
    db.set_submit_mode(uid, 'list')
    superalpha.run(uid, 'u', 'p', seed_plus=10, n=3)
    assert fake.submitted == [], "제출 방식 '목록'인데 자동 제출했다"

    db.set_submit_mode(uid, 'auto')
    fake = _FakeClient()
    fake.harvest_alpha = lambda aid: {'metrics': {}, 'is_status': {
        'pass': [], 'fail': [], 'error': ['X'], 'pending': [], 'warning': []}}
    monkeypatch.setattr(wqb_api, 'WqbApiClient', lambda u, p: fake)
    superalpha.run(uid, 'u', 'p', seed_plus=10, n=2)
    assert fake.submitted == [], '판정 불가(PASS 0) 결과를 냈다'


# ── 대기 슈퍼알파 먼저 (2026-11-16 고급 스프린트 인계) ─────────────────────────

class _QueueClient(_FakeClient):
    """대기 행 제출 결과를 차례로 돌려준다. 시뮬을 부르면 센다(불리면 안 된다)."""

    def __init__(self, results):
        super().__init__()
        self.results = list(results)
        self.descs = {}

    def set_super_description(self, aid, selection, combo):
        self.descs[aid] = (selection, combo)
        self.described = aid
        return True

    def submit_alpha(self, aid, stop_event=None, deadline_s=None):
        self.submitted.append(aid)
        return self.results.pop(0)


def _queue(uid, *wids):
    for w_ in wids:
        db.submit_queue_add(uid, wqb_alpha_id=w_, kind='super',
                            metrics={'sel_desc': f'sel {w_}', 'combo_desc': f'combo {w_}'})


def test_run_submits_queued_superalpha_before_simulating(isolated_db, monkeypatch):
    uid = isolated_db
    import server.wqb_api as wqb_api
    _queue(uid, 'Q1', 'Q2', 'Q3')
    fake = _QueueClient([(False, 'rejected:PROD_CORRELATION(0.88 vs 0.7) (http_403)'),
                         (True, 'submitted')])
    monkeypatch.setattr(wqb_api, 'WqbApiClient', lambda u, p: fake)

    superalpha.run(uid, 'u', 'p', seed_plus=10, n=3)
    assert fake.submitted == ['Q1', 'Q2'] and fake.sim_calls == 0, '대기 행이 있는데 새로 시뮬했다'
    assert fake.descs['Q2'] == ('sel Q2', 'combo Q2'), '저장한 설명을 안 붙였다'
    left = {r['wqb_alpha_id']: r['status'] for r in db.submit_queue_list(uid, include_skipped=True)}
    assert left == {'Q1': 'skipped', 'Q3': 'pending'}   # 성공 Q2 는 내려가고 Q3 은 내일


def test_queued_superalpha_quota_or_timeout_waits_for_tomorrow(isolated_db, monkeypatch):
    uid = isolated_db
    import server.wqb_api as wqb_api
    _queue(uid, 'Q1', 'Q2')
    for msg in ('rejected:SUPER_SUBMISSION(1 vs 1) (http_403)', 'pending_timeout'):
        fake = _QueueClient([(False, msg)])
        monkeypatch.setattr(wqb_api, 'WqbApiClient', lambda u, p: fake)
        superalpha.run(uid, 'u', 'p', seed_plus=10, n=3)
        assert fake.submitted == ['Q1'] and fake.sim_calls == 0
        st = {r['wqb_alpha_id']: r['status'] for r in db.submit_queue_list(uid)}
        assert st == {'Q1': 'pending', 'Q2': 'pending'}, msg


def test_queued_superalpha_transient_failures_keep_the_row(isolated_db, monkeypatch):
    """설명 PATCH 실패(429·5xx 일 수 있다)나 예외는 후보를 버리지도 잠그지도 않는다 — 내일 같은 행부터."""
    uid = isolated_db
    import server.wqb_api as wqb_api
    _queue(uid, 'Q1', 'Q2')
    fake = _QueueClient([])
    fake.set_super_description = lambda aid, s, c: False
    monkeypatch.setattr(wqb_api, 'WqbApiClient', lambda u, p: fake)
    superalpha.run(uid, 'u', 'p', seed_plus=10, n=3)
    assert fake.submitted == [] and fake.sim_calls == 0

    def boom(aid, stop_event=None, deadline_s=None):
        raise RuntimeError('net')
    fake = _QueueClient([])
    fake.submit_alpha = boom
    monkeypatch.setattr(wqb_api, 'WqbApiClient', lambda u, p: fake)
    superalpha.run(uid, 'u', 'p', seed_plus=10, n=3)
    st = {r['wqb_alpha_id']: r['status'] for r in db.submit_queue_list(uid)}
    assert st == {'Q1': 'pending', 'Q2': 'pending'}


def test_already_submitted_queued_superalpha_moves_on(isolated_db, monkeypatch):
    """전날 시간 초과가 실제로 들어갔던 행은 내리고 다음 후보를 낸다(칸이 찼으면 한도 거절로 멈춘다)."""
    uid = isolated_db
    import server.wqb_api as wqb_api
    _queue(uid, 'Q1', 'Q2')
    fake = _QueueClient([(True, 'submitted (이미 제출됨 — stage=OS 확인)'), (True, 'submitted')])
    monkeypatch.setattr(wqb_api, 'WqbApiClient', lambda u, p: fake)
    superalpha.run(uid, 'u', 'p', seed_plus=10, n=3)
    assert fake.submitted == ['Q1', 'Q2'] and db.submit_queue_list(uid) == []
