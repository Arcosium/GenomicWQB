import dataclasses
import datetime as dt
import json
import random
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from server import criteria, db, genome_models as gm, mutation_learn, post_submission as post, reward
from server import worker as w
from server.wqb_api import WqbApiClient
from scripts.monthly_quality import summarize


@pytest.fixture
def user(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'quality.db'))
    monkeypatch.setattr(db, 'SECRET_KEY_PATH', str(tmp_path / 'key'))
    monkeypatch.setattr(db, '_INITIALIZED', False)
    monkeypatch.setattr(db, '_FERNET', None)
    db.init()
    uid = db.upsert_user('quality-test', 'pw', 'test-key')
    yield uid
    db._INITIALIZED = False


def add_alpha(uid, wid='test-alpha', **kw):
    rid = db.start_round(uid, 1)
    return db.insert_alpha(uid, rid, 1, dict(idx=1, code='rank(close)',
        submitted=True, submit_status='submitted',
        metrics={'wqb_alpha_id': wid, 'turnover': .2, 'margin': .0012, 'sharpe': 1.5},
        settings={'region': 'GLB', 'universe': 'TOP3000', 'neutralization': 'INDUSTRY', 'decay': 4},
        genome={'family': 'pv', 'combine': 'spread'}, **kw))


def snapshot(**kw):
    return dict(stage='OS', dateSubmitted='2026-08-01T00:00:00Z',
                os={'sharpe': 1.5, 'returns': .05, 'turnover': .1},
                settings={'universe': 'TOP3000', 'neutralization': 'INDUSTRY', 'decay': 4}, **kw)


NOW = dt.datetime(2026, 11, 5, tzinfo=dt.timezone.utc).timestamp()


def test_margin_targets_and_ht_are_separate():
    assert criteria.good_margin(' asi ') == .0015
    assert reward._margin_term(.2, 504 * .2 * .001, 'GLB') == pytest.approx(1)
    assert reward._margin_term(.2, 504 * .2 * .001, 'ASI') == pytest.approx(2 / 3)
    assert reward._margin_term(.2, 0, 'GLB', margin=.001) == 1
    assert criteria.required_returns(.2) == pytest.approx(504 * .2 * .0003)


@pytest.mark.parametrize('region', ['GLB', 'ASI', 'EUR', 'KOR'])
def test_non_usa_prefers_slow_turnover(region):
    m = dict(region=region, turnover=.28, sharpe=1, returns=.1)
    assert 'ht_gap' not in mutation_learn.categorize([], m)
    assert 'turnover_high' in mutation_learn.categorize([], m)
    assert reward._turnover_term(.12, region=region) > reward._turnover_term(.28, region=region)
    assert gm.turnover_target_for_region(region) == .15


def test_usa_ht_and_legacy_unknown_still_work():
    assert 'ht_gap' in mutation_learn.categorize([], dict(region='USA', turnover=.12))
    assert 'ht_gap' in mutation_learn.categorize([], dict(turnover=.12))
    assert reward._turnover_term(.28, region='USA') == 1
    assert gm.turnover_target_for_region('USA') == .6
    assert gm.decay_for_target_turnover(42, .3, .12) >= 42


def parent():
    return gm.Genome(model='base', family='pv', fields=('close', 'volume', 'returns'),
                     transform_a='rank', transform_b='ts_rank', combine='spread', sign=1,
                     universe='TOP3000', neutralization='INDUSTRY', truncation=.08,
                     decay=4, lookback_a=20, lookback_b=60)


def test_sibling_uses_one_sweep_and_preserves_other_genes():
    g = parent()
    model = gm.BaseGenomeModel(round_num=1, seed_genomes=[g],
                               seed_metrics=[{'_submitted': True, 'turnover': .3, 'region': 'GLB'}])
    plan = model._plan(8)
    assert len(plan) == 8 and sum(x[0] == 'sibling' for x in plan) == 1
    child = model._slow_sibling(g, random.Random(1))
    changed = {k for k, v in dataclasses.asdict(g).items() if dataclasses.asdict(child)[k] != v}
    assert changed <= {'decay', 'lookback_a', 'lookback_b', 'generation'}
    assert child.decay >= g.decay and child.lookback_a > g.lookback_a
    rows = model.generate(8)
    assert len(rows) == 8
    assert sum(r['directive'] == 'slow_sibling' for r in rows) == 1


@pytest.mark.parametrize('metrics,mode', [({'turnover': .3}, 'legacy'),
    ({'_submitted': True, 'turnover': .1}, 'legacy'),
    ({'_submitted': True, 'turnover': .3}, 'escape')])
def test_sibling_requires_success_and_does_not_break_escape(metrics, mode):
    model = gm.BaseGenomeModel(round_num=1, seed_genomes=[parent()], seed_metrics=[metrics], search_mode=mode)
    assert all(op != 'sibling' for op, _, _ in model._plan(8))


@pytest.mark.parametrize('date,hour', [('2026-07-01', 3), ('2026-11-03', 4),
                                      ('2026-03-09', 3), ('2026-11-02', 4)])
def test_release_window_dst(date, hour):
    midnight = dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc)
    begin = midnight.timestamp() + hour * 3600
    assert not w._margin_release_window(begin - 1)
    assert w._margin_release_window(begin)
    assert w._margin_release_window(begin + 3599)
    assert not w._margin_release_window(begin + 3600)


def test_gate_defers_then_releases_low_margin(user, monkeypatch):
    worker = w.Worker.__new__(w.Worker)
    worker.user_id = user
    worker._stop_event = threading.Event()
    monkeypatch.setattr(worker, '_submitted_today', lambda: 0)
    monkeypatch.setattr(w, '_pp_month_closed', lambda _: False)
    monkeypatch.setattr(w.run_config, 'get_submit_hold_until', lambda: 0)
    monkeypatch.setattr(w, '_margin_release_window', lambda: False)
    m = {'wqb_alpha_id': 'low', 'sharpe': 2, 'margin': .0005, 'region': 'GLB'}
    assert worker._submit_gate(m, None, fail_items=[], code='rank(close)') == (False, 'low_margin→queued')
    assert db.submit_queue_list(user)[0]['kind'] == 'budget'
    monkeypatch.setattr(w, '_margin_release_window', lambda: True)
    assert worker._submit_gate(m, None, fail_items=[], code='rank(close)') == (True, '')
    db.set_submit_mode(user, 'list')
    assert worker._submit_gate(m, None, fail_items=[])[0] is False


def test_drain_skips_low_margin_until_window(user, monkeypatch):
    worker = w.Worker.__new__(w.Worker)
    worker.user_id = user
    db.submit_queue_add(user, wqb_alpha_id='low', kind='budget', metrics={'margin': .0001})
    monkeypatch.setattr(w, '_margin_release_window', lambda: False)
    monkeypatch.setattr(worker, '_submitted_today', lambda: pytest.fail('low margin reached budget/API'))
    assert worker._drain_one(1, 'u', 'p') is None


def test_pyramid_priority_stays_inside_margin_and_fresh_tiers(monkeypatch):
    monkeypatch.setattr(w, '_datasets_used_by_code', lambda _: {'ds'})
    def row(i, cell, margin=.0012):
        return {'id': i, 'code': 'rank(close)', 'metrics': {'margin': margin, 'pyramids': cell}}
    counts = {'GLB/D1/PV': 3}
    full, short, thin = row(1, 'GLB/D1/PV'), row(2, 'GLB/D1/NEWS'), row(3, 'GLB/D1/NEWS', .0001)
    assert max([full, short, thin], key=lambda r: w._drain_key(r, set(), counts)) is short
    assert not w._drain_key(short, set(), None)[2]


def test_os_api_reads_without_auth_or_is_overwrite():
    data = snapshot()
    client = WqbApiClient('u', 'p', session_file=False)
    client.session = SimpleNamespace(get=lambda *a, **k: SimpleNamespace(ok=True, json=lambda: data))
    client.authenticate = lambda: pytest.fail('must not reauthenticate')
    assert client.harvest_post_submission('a')['os']['sharpe'] == 1.5


@pytest.mark.parametrize('os', [{}, {'sharpe': None}, {'sharpe': float('nan')},
    {'sharpe': 1, 'returns': .1, 'turnover': 0}])
def test_empty_os_is_not_a_loss(os):
    s = snapshot(); s['os'] = os
    assert post.os_score(s, NOW) is None


def test_os_maturity_and_signed_feedback():
    s = snapshot(); s['dateSubmitted'] = '2026-11-01T00:00:00Z'
    assert post.os_score(s, NOW) is None
    s = snapshot(); s['os']['sharpe'] = -2
    assert post.os_score(s, NOW) < 0


def test_os_refresh_idempotent_and_user_scoped(user):
    aid = add_alpha(user)
    other = db.upsert_user('other-test', 'pw', 'test-key')
    add_alpha(other, 'other-alpha')
    client = SimpleNamespace(harvest_post_submission=lambda _: snapshot())
    stop = SimpleNamespace(is_set=lambda: False, wait=lambda _: False)
    db.bandit_update(user, 'family:pv', .6, 1, passed=True)
    before = db.bandit_stats(user)
    assert post.refresh(user, client, stop, now=NOW) == 1
    bonus = post.bandit_adjustments(user, now=NOW)
    assert 0 < bonus['family:pv'] < post.OS_WEIGHT
    assert post.refresh(user, client, stop, now=NOW) == 0
    assert post.refresh(user, client, stop, now=NOW+86401) == 1
    assert post.bandit_adjustments(user, now=NOW+86401) == bonus
    assert db.bandit_stats(user) == before
    assert post.bandit_adjustments(other, now=NOW) == {}
    assert post.bandit_adjustments(user, now=NOW+9*86400) == {}
    with db._connect() as conn:
        assert json.loads(conn.execute('SELECT metrics FROM alphas WHERE id=?', (aid,)).fetchone()[0])['sharpe'] == 1.5


def test_failed_fetch_keeps_previous_observation_and_retries(user):
    add_alpha(user)
    stop = SimpleNamespace(is_set=lambda: False, wait=lambda _: False)
    assert post.refresh(user, SimpleNamespace(harvest_post_submission=lambda _: None), stop, now=NOW) == 0
    assert len(db.post_submission_candidates(user, NOW)) == 1


def test_monthly_uses_submission_date_deduplicates_and_reports_missing(user):
    aid = add_alpha(user)
    add_alpha(user)  # same remote id must count once
    s = snapshot(); s['dateSubmitted'] = '2026-10-15T10:00:00Z'
    db.save_post_submission(user, aid, 'test-alpha', s, [], .5, NOW)
    add_alpha(user, 'undated')
    with sqlite3.connect(db.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        report = summarize(conn, user, '2026-10')
    assert report['submitted_count'] == 1
    assert report['turnover_pct'] == 20
    assert report['margin_bps'] == pytest.approx(12)
    assert report['measurement_counts']['drawdown'] == 0
    assert report['drawdown_pct'] is None
    assert report['undated_submitted_count'] == 1
    assert report['power_pool_combined_sharpe'] is None
    assert report['date_basis'] == {'api_submission_date': 1}


def test_v9_upgrade_does_not_replay_submission_backfill(user):
    aid = add_alpha(user)
    with db._connect() as conn:
        conn.execute("UPDATE alphas SET submitted=0, submit_status='submitted' WHERE id=?", (aid,))
        conn.execute('DROP TABLE post_submission')
        conn.execute('PRAGMA user_version=9')
    db._INITIALIZED = False
    db.init()
    with db._connect() as conn:
        assert conn.execute('PRAGMA user_version').fetchone()[0] == 10
        assert conn.execute('SELECT submitted FROM alphas WHERE id=?', (aid,)).fetchone()[0] == 0
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='post_submission'").fetchone()


def test_os_candidates_ignore_malformed_metrics_and_deduplicate(user):
    add_alpha(user)
    newest = add_alpha(user)
    bad = add_alpha(user, 'bad')
    with db._connect() as conn:
        conn.execute("UPDATE alphas SET metrics='broken' WHERE id=?", (bad,))
    rows = db.post_submission_candidates(user, NOW)
    assert len(rows) == 1 and rows[0]['id'] == newest


def test_seed_rows_keep_submission_and_region(user):
    add_alpha(user)
    seeds = db.hall_of_fame_seeds(user)
    assert seeds[0]['submitted'] == 1
    assert seeds[0]['metrics']['region'] == 'GLB'


def test_monthly_local_confirmation_can_cross_simulation_month(user):
    aid = add_alpha(user)
    submitted = dt.datetime(2026, 10, 15, tzinfo=dt.timezone.utc).timestamp()
    db.record_submit_attempt(user, 1, 1, 'rank(close)', True, 'submitted')
    with db._connect() as conn:
        conn.execute('UPDATE submit_attempts SET ts=? WHERE user_id=?', (submitted, user))
        conn.execute('UPDATE alphas SET ts=? WHERE id=?', (submitted - 40 * 86400, aid))
        report = summarize(conn, user, '2026-10')
    assert report['submitted_count'] == 1
    assert report['date_basis'] == {'local_confirmation_date': 1}


def test_rc_sibling_keeps_measured_decay_in_final_settings():
    rows = gm.generate_population(account_type='research_consultant', round_num=1, n=8,
        seed_genomes=[parent()], seed_metrics=[{'_submitted': True, 'turnover': .4, 'region': 'GLB'}])
    sibling = next(r for r in rows if r['directive'] == 'slow_sibling')
    assert sibling['genome']['decay'] > 8
    assert int(sibling['settings']['decay']) == sibling['genome']['decay']
    assert 16 in gm.ResearchConsultantGenomeModel.decays


def test_rc_keeps_existing_decorrelation_decay():
    g = dataclasses.replace(parent(), decay=42)
    model = gm.ResearchConsultantGenomeModel(round_num=1, parent_genome=g)
    assert model.parent.decay == 42
    assert model._constrain(g, random.Random(0)).decay == 42


@pytest.mark.parametrize('region,expected', [('USA', 'signal'), (None, 'signal'), ('GLB', 'turnover_high')])
def test_directed_mutation_preserves_usa_signal(monkeypatch, region, expected):
    from server import submission_feedback
    monkeypatch.setattr(submission_feedback, 'bottleneck', lambda _: {'name': 'LOW_SHARPE'})
    model = gm.BaseGenomeModel(round_num=1, seed_genomes=[parent()],
        seed_metrics=[dict(region=region, turnover=.28 if region == 'GLB' else .12)])
    model.search_mode = 'directed'
    model._mutate(parent(), random.Random(1))
    assert model._last_directive == mutation_learn.RULE_DIRECTIVE[expected]


@pytest.mark.parametrize('status,data', [(404, None), (410, None), (200, []), (200, ValueError())])
def test_individual_os_failure_does_not_block_following_alpha(user, status, data):
    add_alpha(user, 'bad'); add_alpha(user, 'good')
    client = WqbApiClient('u', 'p', session_file=False)
    def get(url, **kwargs):
        def payload():
            if isinstance(data, Exception):
                raise data
            return data
        return (SimpleNamespace(ok=status == 200, status_code=status, json=payload)
                if url.endswith('/bad') else SimpleNamespace(ok=True, json=snapshot))
    client.session = SimpleNamespace(get=get)
    stop = SimpleNamespace(is_set=lambda: False, wait=lambda _: False)
    assert post.refresh(user, client, stop, now=NOW) == 2
    assert db.post_submission_candidates(user, NOW - 86400) == []
    assert post.bandit_adjustments(user, now=NOW)['family:pv'] > 0


@pytest.mark.parametrize('status', [401, 403, 429, 500])
def test_systemic_os_failure_stops_batch(status):
    client = WqbApiClient('u', 'p', session_file=False)
    client.session = SimpleNamespace(get=lambda *a, **k: SimpleNamespace(ok=False, status_code=status))
    assert client.harvest_post_submission('bad') is None


def test_stuck_retry_skips_queued_low_margin(monkeypatch):
    worker = object.__new__(w.Worker)
    worker.user_id = 1
    rows = [dict(id=i, metrics={'wqb_alpha_id': str(i)}) for i in (1, 2)]
    monkeypatch.setattr(w._db, 'stuck_submits', lambda _: rows)
    seen = []
    def gate(metrics, *a, **kw):
        seen.append(metrics['wqb_alpha_id'])
        return False, 'low_margin→queued' if len(seen) == 1 else 'budget'
    monkeypatch.setattr(worker, '_submit_gate', gate)
    monkeypatch.setattr(worker, '_log_quiet', lambda *a: None)
    worker._retry_stuck_submits(1, 'u', 'p')
    assert seen == ['1', '2']
