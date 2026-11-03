#!/usr/bin/env python3
"""월간 제출 품질을 읽기 전용 SQLite 연결로 집계한다.

python3 scripts/monthly_quality.py --month 2026-10 --user-id 1 [--baseline private.json]
제출일을 알 수 없는 행은 생성일로 대체하지 않고 제외 건수로 표시한다.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import alpha_ast, criteria
from server.post_submission import number, submitted_ts


def object_json(value):
    try:
        result = json.loads(value or '{}')
        return result if isinstance(result, dict) else {}
    except (TypeError, ValueError):
        return {}


def summarize(conn, user_id, month, timezone='America/New_York'):
    tz = ZoneInfo(timezone)
    start = datetime.strptime(month, '%Y-%m').replace(tzinfo=tz)
    end = start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)
    rows = [dict(r) for r in conn.execute(
        'SELECT * FROM alphas WHERE user_id=? AND submitted=1 ORDER BY id DESC', (user_id,))]
    snapshots = {}
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name='post_submission'").fetchone():
        snapshots = {r['wqb_alpha_id']: object_json(r['snapshot']) for r in conn.execute(
            'SELECT wqb_alpha_id,snapshot FROM post_submission WHERE user_id=?', (user_id,))}
    attempts = defaultdict(list)
    for a in conn.execute('SELECT code,round_num,idx,ts FROM submit_attempts '
                          'WHERE user_id=? AND submitted=1 ORDER BY ts', (user_id,)):
        attempts[a['code']].append(dict(a))
    identities = defaultdict(set)
    for row in rows:
        wid = object_json(row['metrics']).get('wqb_alpha_id')
        identities[row['code']].add(wid or f"local:{row['id']}")
    seen, selected, unknown = set(), [], 0
    bases = Counter()
    for row in rows:
        m = object_json(row['metrics'])
        wid = m.get('wqb_alpha_id') or f"local:{row['id']}"
        if wid in seen:
            continue
        seen.add(wid)
        snapshot = snapshots.get(wid, {})
        ts = submitted_ts(snapshot.get('dateSubmitted') or m.get('dateSubmitted'))
        basis = 'api_submission_date'
        if ts is None:
            matches = [a for a in attempts[row['code']]
                       if a['round_num'] == row['round_num'] and a['idx'] == row['idx']]
            if not matches and len(identities[row['code']]) == 1:
                matches = attempts[row['code']]
            ts = min((a['ts'] for a in matches), default=None)
            basis = 'local_confirmation_date'
        if ts is None:
            unknown += 1
            continue
        if not start.timestamp() <= ts < end.timestamp():
            continue
        selected.append((row, m, snapshot))
        bases[basis] += 1
    vals = defaultdict(list)
    fields, operators, cells = set(), set(), Counter()
    os_count = 0
    for row, m, snapshot in selected:
        for key in ('turnover', 'margin', 'sharpe', 'drawdown'):
            v = number(criteria.margin(m)) if key == 'margin' else number(criteria._f(m.get(key)))
            if v is not None:
                vals[key].append(v)
        fields.update(alpha_ast.fields_used(row['code']))
        operators.update(alpha_ast.operators_used(row['code']))
        cells.update(set(x.strip() for x in str(m.get('pyramids') or '').split(',') if x.strip()))
        if number((snapshot.get('os') or {}).get('sharpe')) is not None:
            os_count += 1
    def mean(key, scale=1):
        return sum(vals[key]) / len(vals[key]) * scale if vals[key] else None
    result = dict(month=month, timezone=timezone, submitted_count=len(selected),
                  turnover_pct=mean('turnover', 100), margin_bps=mean('margin', 10000),
                  sharpe=mean('sharpe'), drawdown_pct=mean('drawdown', 100),
                  margin_10bps_pct=(100 * sum(x >= .001 for x in vals['margin']) / len(vals['margin'])
                                    if vals['margin'] else None),
                  unique_fields=len(fields), unique_operators=len(operators),
                  monthly_pyramid_cells_ge3=sum(n >= 3 for n in cells.values()),
                  os_coverage_pct=100 * os_count / len(selected) if selected else None,
                  daily_submissions=len(selected) / (end.date() - start.date()).days,
                  measurement_counts={k: len(vals[k]) for k in ('turnover', 'margin', 'sharpe', 'drawdown')},
                  date_basis=dict(bases), undated_submitted_count=unknown,
                  power_pool_combined_sharpe=None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=Path(os.environ.get('HYFE_DB_PATH') or
                        Path(__file__).resolve().parents[1] / 'data/hyfe_iqc.db'))
    parser.add_argument('--user-id', type=int, required=True)
    parser.add_argument('--month', required=True)
    parser.add_argument('--timezone', default='America/New_York')
    parser.add_argument('--baseline', type=Path, help='같은 키를 쓰는 개인 기준값 JSON')
    args = parser.parse_args()
    conn = sqlite3.connect(args.db.resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        result = summarize(conn, args.user_id, args.month, args.timezone)
    finally:
        conn.close()
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        result['baseline'] = baseline
        result['difference'] = {k: result[k] - v for k, v in baseline.items()
                                if number(v) is not None and number(result.get(k)) is not None}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
