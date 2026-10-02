"""Submission-time observations for research allocation, never submission gates."""
from __future__ import annotations

import math
import re

METRICS = {
    'LOW_SHARPE': 'sharpe', 'LOW_FITNESS': 'fitness',
    'LOW_GLB_AMER_SHARPE': 'glb_amer_sharpe',
    'LOW_GLB_EMEA_SHARPE': 'glb_emea_sharpe',
    'LOW_GLB_APAC_SHARPE': 'glb_apac_sharpe',
    'LOW_SUB_UNIVERSE_SHARPE': 'sub_universe_sharpe',
    'IS_LADDER_SHARPE': 'ladder_sharpe', 'LOW_2Y_SHARPE': 'sharpe_2y',
    'HIGH_TURNOVER': 'turnover', 'LOW_TURNOVER': 'turnover',
    'CONCENTRATED_WEIGHT': 'weight_concentration',
    'PROD_CORRELATION': 'prod_correlation',
    'SELF_CORRELATION': 'self_correlation',
}
UPPER_LIMITS = {'HIGH_TURNOVER', 'CONCENTRATED_WEIGHT',
                'PROD_CORRELATION', 'SELF_CORRELATION'}
STABILITY = ('IS_LADDER_SHARPE', 'LOW_2Y_SHARPE')
_VALUE = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
_FAIL = re.compile(r'([A-Z][A-Z0-9_]+)\(\s*(' + _VALUE
                   + r')\s+vs\s+(' + _VALUE + r')\s*\)')


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def compact_checks(body) -> list[dict]:
    if not isinstance(body, dict):
        return []
    checks = (body.get('is') or {}).get('checks') or []
    return [{k: c[k] for k in ('name', 'result', 'value', 'limit',
                              'year', 'startDate', 'endDate') if k in c}
            for c in checks if isinstance(c, dict) and c.get('name')]


def observations(row: dict) -> list[dict]:
    metrics = row.get('metrics') or row.get('parent_metrics') or {}
    checks = [dict(c) for c in metrics.get('_submit_checks', [])
              if isinstance(c, dict) and c.get('name') and
              (c.get('name') in METRICS or c.get('result') in ('FAIL', 'ERROR'))]
    # Old rows retain numeric submission evidence in the status string. Do not
    # invent missing/truncated checks or infer PASS from their absence.
    seen = {c['name'] for c in checks}
    status = str(row.get('submit_status') or '')
    if status.startswith('rejected:'):
        for name, value, limit in _FAIL.findall(status):
            parsed = {'name': name, 'result': 'FAIL',
                      'value': float(value), 'limit': float(limit)}
            if name not in seen:
                checks.append(parsed)
                seen.add(name)
            else:
                index = next(i for i, c in enumerate(checks) if c['name'] == name)
                old = checks[index]
                if (number(old.get('value')), number(old.get('limit')), old.get('result')) != (
                        parsed['value'], parsed['limit'], 'FAIL'):
                    # A later queue attempt can update an older stored snapshot.
                    checks[index] = parsed
    return checks


def effective_metrics(row: dict) -> dict:
    metrics = dict(row.get('metrics') or row.get('parent_metrics') or {})
    for c in observations(row):
        value = number(c.get('value'))
        if value is not None and c['name'] in METRICS:
            metrics[METRICS[c['name']]] = value
            metrics[METRICS[c['name']] + '_cutoff'] = c.get('limit')
    return metrics


def deficit(check: dict) -> float | None:
    if check.get('name') not in METRICS:
        return None
    value, limit = number(check.get('value')), number(check.get('limit'))
    if value is None or limit is None:
        return None
    gap = value - limit if check['name'] in UPPER_LIMITS else limit - value
    return max(0.0, gap / max(abs(limit), 0.01))


def bottleneck(row: dict) -> dict | None:
    failed = [c for c in observations(row) if c.get('result') == 'FAIL'
              and deficit(c) is not None]
    return max(failed, key=lambda c: deficit(c), default=None)


def failure_descriptions(row: dict) -> list[str]:
    failed = [c for c in observations(row) if c.get('result') in ('FAIL', 'ERROR')]
    failed.sort(key=lambda c: deficit(c) or 0.0, reverse=True)
    return [f"{c['name']}({c['value']} vs {c['limit']})"
            if number(c.get('value')) is not None and number(c.get('limit')) is not None
            else f"{c['name']}={c.get('result')}" for c in failed]


def stability_improved(parent: dict, child: dict, epsilon=0.02):
    """Compare the same temporal check; overall Sharpe is never a proxy."""
    pc = {c['name']: c for c in observations(parent) if c['name'] in STABILITY}
    cc = {c['name']: c for c in observations(child) if c['name'] in STABILITY}
    for name in STABILITY:
        if name not in pc or name not in cc:
            continue
        p, c = pc[name], cc[name]
        if any(p.get(k) != c.get(k) for k in ('year', 'startDate', 'endDate')):
            continue
        if c.get('result') not in ('PASS', 'FAIL') or p.get('result') not in ('PASS', 'FAIL'):
            continue
        if p.get('result') == 'FAIL' and c.get('result') == 'PASS':
            return True
        pv, cv = deficit(p), deficit(c)
        if pv is not None and cv is not None:
            return cv < pv - epsilon
    # Historical snapshots have metrics but no full check body. Require the
    # same actual temporal metric on both sides and no explicit window conflict.
    if pc or cc:
        return None
    pm, cm = parent.get('metrics') or {}, child.get('metrics') or {}
    for key in ('ladder_sharpe', 'sharpe_2y'):
        pv, cv = number(pm.get(key)), number(cm.get(key))
        if pv is not None and cv is not None:
            return cv > pv + epsilon * max(abs(pv), 1e-9)
    return None
