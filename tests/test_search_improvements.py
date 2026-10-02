"""Regression coverage for the 2.2 research allocation and submission feedback."""
import random

import pytest

from server import (field_types, genome_models as gm, mutation_learn,
                    research_v2 as policy, submission_feedback as feedback,
                    wqb_backend)


def observed(value=1.0, *, limit=1.58, name='IS_LADDER_SHARPE', result='FAIL', **extra):
    return {'name': name, 'result': result, 'value': value, 'limit': limit, **extra}


def row(value=1.0, **extra):
    return {'code': 'rank(close)', 'metrics': {'sharpe': 3.1, 'fitness': 1.2,
            '_submit_checks': [observed(value)]},
            'submit_status': f'rejected:IS_LADDER_SHARPE({value} vs 1.58)', **extra}


@pytest.mark.parametrize('mode', ['exploit', 'explore'])
def test_four_single_axis_slots_in_fourteen_are_spread_across_parents(mode):
    gm.set_constraint(None)
    seeds = gm.generate_population(account_type='research_consultant',
                                    round_num=123, forced_delay='1', n=5)
    children = gm.generate_population(account_type='research_consultant',
        round_num=234, forced_delay='1', n=14,
        seed_genomes=[s['genome'] for s in seeds], seed_alpha_ids=[1, 2, 3, 4, 5],
        seed_metrics=[{'sharpe': 1.5, 'fitness': 0.9}] * 5,
        search_mode=mode)
    sweeps = [s for s in children if s['origin'] == 'sweep']
    assert len(children) == 14 and len(sweeps) == 4
    assert len({s['parent_alpha_id'] for s in sweeps}) == 4
    assert all(len(s['genes_changed']) == 1 for s in sweeps)
    assert sum(s['origin'] == 'random' for s in children) <= len(children) * 0.30


def test_parent_turnover_is_used_in_its_own_sweep():
    gm.set_constraint(None)
    seed = gm.generate_population(account_type='research_consultant',
                                  round_num=124, forced_delay='1', n=1)[0]['genome']
    seed['decay'] = 2
    model = gm.ResearchConsultantGenomeModel(round_num=2, forced_delay='1',
        seed_genomes=[seed], seed_metrics=[{'turnover': 1.0}], search_mode='exploit')
    child = model._sweep(model.seeds[0], 1, 1)
    assert child.decay > 2
    assert child.transform_a == model.seeds[0].transform_a


def test_stability_failure_reaches_targeted_and_single_axis_mutation():
    gm.set_constraint(None)
    seed = gm.generate_population(account_type='research_consultant',
                                  round_num=125, forced_delay='1', n=1)[0]['genome']
    children = gm.generate_population(account_type='research_consultant',
        round_num=2, forced_delay='1', n=14, parent_genome=seed,
        parent_metrics=row()['metrics'], fail_items=['IS_LADDER_SHARPE'],
        search_mode='exploit')
    sweeps = [s for s in children if s['origin'] == 'sweep']
    assert len(sweeps) == 4
    assert all(len(s['genes_changed']) == 1 for s in sweeps)
    assert all(set(s['genes_changed']) <= {'lookback_a', 'lookback_b', 'winsor_std', 'regime'}
               for s in sweeps)
    assert any(s['directive'] == 'robustify' for s in children)


def test_all_submit_failures_survive_and_replace_provisional_check_states():
    checks = [observed(0.1, name=n) for n in list(feedback.METRICS)[:8]]
    result = {'submit_status': 'rejected:some checks',
              'metrics': {'sharpe': 3.1, '_submit_checks': checks},
              'is_status': {'pass': [{'name': 'IS_LADDER_SHARPE'}]}}
    policy.promote_submit_evidence(result)
    assert len(feedback.failure_descriptions(result)) == 8
    assert 'IS_LADDER_SHARPE' in {c['name'] for c in result['fail_items']}
    assert 'IS_LADDER_SHARPE' not in {c['name'] for c in result['is_status']['pass']}
    assert all(isinstance(c['value'], str) for c in result['fail_items'])
    assert result['metrics']['sharpe'] == 3.1  # original simulation remains intact
    assert feedback.effective_metrics(result)['sharpe'] == 0.1


def test_weak_temporal_result_loses_priority_despite_high_overall_sharpe():
    good = row(1.5)
    bad = row(0.4)
    bad['metrics']['sharpe'] = 3.2
    assert policy.candidate_priority(good) > policy.candidate_priority(bad)


def test_stability_learning_does_not_reward_an_overall_sharpe_increase():
    parent = row(1.0, fail_items=['IS_LADDER_SHARPE'], pass_count=7)
    child = row(0.9, fail_items=['IS_LADDER_SHARPE'], pass_count=7, directive='robustify')
    child['metrics']['sharpe'] = 5.0
    assert mutation_learn.outcome_observations(parent, child) == [('stability', 'robustify', False)]
    child = row(1.2, fail_items=['IS_LADDER_SHARPE'], pass_count=7, directive='robustify')
    assert mutation_learn.outcome_observations(parent, child) == [('stability', 'robustify', True)]


def test_unknown_or_different_temporal_window_is_not_a_learning_win():
    parent = row(1.0, fail_items=['IS_LADDER_SHARPE'], pass_count=7)
    child = {'metrics': {'sharpe': 10}, 'pass_count': 8, 'directive': 'robustify'}
    assert mutation_learn.outcome_observations(parent, child) == [('stability', 'robustify', False)]
    parent['metrics']['_submit_checks'][0]['year'] = 2
    child = row(1.5)
    child['metrics']['_submit_checks'][0]['year'] = 3
    assert feedback.stability_improved(parent, child) is None


def test_stalled_lineage_releases_focus_and_recovers_after_progress():
    recent = [row(1.0, id=i, ts=i) for i in range(1, 13)]
    p = policy.build_lineage_policy(recent)
    assert not policy.focus_allowed(recent[-1], p)[0]
    assert policy.focus_allowed({'code': 'rank(volume)',
                                 'metrics': {'sharpe': 1.5, 'fitness': 0.9}}, p)[0]
    assert not policy.focus_allowed({'parent_code': 'rank(close)'}, p)[0]
    recent[-1] = row(1.3, id=12, ts=12)
    assert policy.focus_allowed(recent[-1], policy.build_lineage_policy(recent))[0]
    recent[-1] = row(1.0, id=12, ts=12, submit_status='submitted')
    assert policy.focus_allowed(recent[-1], policy.build_lineage_policy(recent))[0]


def test_cached_rows_do_not_create_stagnation():
    recent = [row(1.0, id=i, cached=True) for i in range(20)]
    assert not policy.build_lineage_policy(recent)['stalled_lineages']


@pytest.mark.parametrize('code', [
    'rank(currency)', 'ts_delta(exchange, 5)', 'ts_zscore(country, 20)',
    'ts_corr(close, ts_backfill(currency, 10), 20)',
    'g = country; rank(g)', 'rank(densify(country))',
    'group_rank(currency, sector)',
])
def test_group_values_cannot_be_numeric_signal_inputs(code):
    assert field_types.numeric_input_reason(code)


@pytest.mark.parametrize('code', [
    'group_rank(close,country)', 'group_neutralize(ts_delta(close,5),sector)',
    'g = densify(country); group_rank(close,g)',
    'rank(vec_avg(unknown_vector_field))', 'rank(unknown_type)',
    'group_rank(close,bucket(rank(cap),range="0,1,0.1"))',
])
def test_group_arguments_and_unknown_types_remain_usable(code):
    assert field_types.numeric_input_reason(code) is None


def test_numeric_generation_excludes_groups_even_from_legacy_seeds(monkeypatch):
    gm.set_constraint(None)
    monkeypatch.setitem(gm.SHARED_DATASETS, 'pv', ['currency', 'country', 'close', 'open', 'volume'])
    assert not (set(gm._pick_fields(random.Random(1), 'pv', set(), '1')) & {'currency', 'country'})
    seed = gm.generate_population(account_type='research_consultant',
                                  round_num=125, forced_delay='1', n=1)[0]['genome']
    seed.update(fields=('currency', 'country', 'exchange'))
    model = gm.ResearchConsultantGenomeModel(round_num=1, forced_delay='1')
    child = model._constrain(gm._coerce_genome(seed), random.Random(1))
    assert not set(child.fields) & field_types.BUILTIN_GROUPS


def test_backend_blocks_known_type_error_before_spending_a_simulation():
    backend = wqb_backend.ApiBackend('fake', 'fake', client=object())
    backend._submit_with_retry = lambda *a: pytest.fail('invalid input reached WQB')
    result = backend._run_one({'idx': 1, 'code': 'rank(currency)'}, '1', None, None)
    assert result['error_text'].startswith('preflight type:')


def test_region_constraint_does_not_reintroduce_group_fields(monkeypatch):
    from server import constraint_spec
    gm.set_constraint(None)
    monkeypatch.setattr(gm, '_ACTIVE_CONSTRAINT', constraint_spec.parse('region=GLB & delay=1'))
    pool = ('country', 'currency', 'exchange', 'close', 'open', 'volume')
    monkeypatch.setattr(gm, '_REGION_DATASETS', {'pv': pool})
    monkeypatch.setattr(gm, '_REGION_FULL_FIELDS', frozenset(pool))
    for i in range(30):
        d = {'family': 'pv', 'fields': (f'old_{i}', 'old_b', 'old_c'),
             'neutralization': 'MARKET', 'decay': i, 'transform_a': 'rank'}
        gm._apply_constraint(d)
        assert not set(d['fields']) & field_types.BUILTIN_GROUPS


def test_api_preserves_full_submission_checks_and_clears_them_between_attempts():
    from server.wqb_api import WqbApiClient
    body = {'is': {'checks': [observed(0.9, year=2, startDate='2024-01-01'),
                             observed(0.0, name='LOW_DURATION')]}}
    response = type('Response', (), {'status_code': 403, 'headers': {}, 'text': '',
                                      'json': lambda self: body})()
    client = WqbApiClient.__new__(WqbApiClient)
    client._ensure_auth = lambda: True
    client._verify_submitted = lambda aid: False
    client.session = type('Session', (), {'post': lambda *a, **k: response})()
    ok, status = client.submit_alpha('A1')
    assert not ok and status.startswith('rejected:')
    assert client._last_submit_checks == body['is']['checks']
    assert client._last_submit_alpha_id == 'A1'
    client.submit_alpha('')
    assert client._last_submit_checks == []


def test_new_queue_rejection_replaces_a_stale_observation():
    r = row(1.0)
    r['submit_status'] = 'rejected:IS_LADDER_SHARPE(1.3 vs 1.58)'
    assert feedback.effective_metrics(r)['ladder_sharpe'] == 1.3


def test_pending_stability_is_not_counted_as_progress():
    parent = row(1.0)
    child = row(1.5, submit_status='submit_pending_timeout')
    child['metrics']['_submit_checks'][0]['result'] = 'PENDING'
    assert feedback.stability_improved(parent, child) is None


def test_plateau_and_stricter_checks_cannot_expand_exploration_cadence():
    recent = [row(0.2, id=i, ts=i) for i in range(40)]
    p = {'near_miss_count': 0, 'quarantined': []}
    modes = [policy.choose_search_mode(i, recent, policy=p)[0] for i in range(1, 41)]
    assert modes.count('explore') == 10
    assert modes.count('exploit') == 30
    assert all(policy.choose_search_mode(i, recent, focus_code='rank(close)',
                                        policy=p)[0] == 'exploit' for i in range(1, 41))


@pytest.mark.parametrize('mode', ['exploit', 'explore'])
def test_focus_never_generates_unrelated_random_children(mode):
    gm.set_constraint(None)
    seed = gm.generate_population(account_type='research_consultant',
                                  round_num=125, forced_delay='1', n=1)[0]['genome']
    children = gm.generate_population(account_type='research_consultant',
        round_num=444, forced_delay='1', n=14, parent_genome=seed,
        parent_alpha_id=123, parent_metrics={'sharpe': 1.5, 'fitness': 0.9},
        fail_items=['LOW_GLB_EMEA_SHARPE'], search_mode=mode)
    assert len(children) == 14
    assert all(s['parent_alpha_id'] == 123 and s['origin'] != 'random' for s in children)


def test_sweep_budget_goes_only_to_measured_signal():
    gm.set_constraint(None)
    seeds = gm.generate_population(account_type='research_consultant',
                                    round_num=123, forced_delay='1', n=5)
    metrics = [{'sharpe': -0.5, 'fitness': -0.1}, {},
               {'sharpe': 0.5, 'fitness': 0.3},
               {'sharpe': 1.5, 'fitness': 0.9},
               {'sharpe': 2, 'fitness': 2, '_submit_checks': [
                   observed(36, limit=500, name='LOW_DURATION')]}]
    children = gm.generate_population(account_type='research_consultant',
        round_num=234, forced_delay='1', n=14,
        seed_genomes=[s['genome'] for s in seeds], seed_alpha_ids=[1, 2, 3, 4, 5],
        seed_metrics=metrics, search_mode='exploit')
    sweeps = [s for s in children if s['origin'] == 'sweep']
    assert sweeps and all(s['parent_alpha_id'] == 4 for s in sweeps)
    for m in metrics[:3] + metrics[4:]:
        assert not policy.focus_allowed({'code': 'rank(close)', 'metrics': m})[0]


def test_exhausted_sweeps_refill_with_local_children_of_same_parent():
    gm.set_constraint(None)
    seed = gm.generate_population(account_type='research_consultant',
                                  round_num=125, forced_delay='1', n=1)[0]['genome']
    children = gm.generate_population(account_type='research_consultant',
        round_num=444, forced_delay='1', n=14, parent_genome=seed,
        parent_alpha_id=123, parent_metrics={'sharpe': 1.5, 'fitness': 0.9},
        fail_items=['LOW_GLB_EMEA_SHARPE'], search_mode='exploit',
        accept_candidate=lambda s: s['origin'] != 'sweep')
    assert len(children) == 14
    assert all(s['parent_alpha_id'] == 123 for s in children)
    assert not any(s['origin'] in ('random', 'sweep') for s in children)


def test_final_budget_excludes_cache_losses_and_specs_from_random_allowance():
    # Most local proposals were duplicates. Specs are independently allocated,
    # so they must not inflate the allowance for unrelated random candidates.
    candidates = ([{'origin': 'local', 'idx': i} for i in range(3)]
                  + [{'origin': 'spec', 'idx': 10 + i} for i in range(8)]
                  + [{'origin': 'random', 'idx': 20 + i} for i in range(4)])
    kept, deferred = policy.limit_random_candidates(candidates)
    assert len(deferred) == 3
    assert sum(s['origin'] == 'random' for s in kept) == 1
    assert sum(s['origin'] == 'spec' for s in kept) == 8
    kept, _ = policy.limit_random_candidates(candidates, focus=True)
    assert all(s['origin'] != 'random' for s in kept)
    assert policy.limit_random_candidates(candidates, bootstrap=True) == (candidates, [])
