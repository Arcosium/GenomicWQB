#!/usr/bin/env python3
"""데이터셋 하나의 전체 필드를 단일 필드 템플릿 하나로 훑는다.

왜 — 2026-10 리포트 지적: 제출 47건이 고유 필드 29개(전 세계 하위권)로 거의 같은 재료였다.
Genius 타이브레이커 '총 고유 필드 수'를 늘리려면 새 데이터셋의 필드를 통째로 재 봐야 한다.
한국 중급 스프린트 과제 2(데이터셋 발굴)와 같은 작업이라 결과 요약도 그 양식 항목을 낸다.

워커와 시뮬 슬롯을 나눠 쓴다(campaign.run 이 429 를 기다린다). 인증 POST 는 하지 않는다 —
워커의 저장 세션이 죽어 있으면 그대로 멈춘다(얼굴 인증 재무장 방지).

    python3.12 scripts/scan_dataset.py ASI MINVOL1M multi_source_model \
        --template 'group_rank(ts_backfill({f}, 60), country)' --neut INDUSTRY --decay 4

결과: _workspace/scan_<region>_<dataset>.json — 필드별 S·fitness·회전율·margin·커버리지·알파 id.
VECTOR 필드는 {f} 자리에 vec_avg(필드)가 들어간다.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server import campaign, wqb_api, wqb_data_service as wds  # noqa: E402

PROMISING_SHARPE = 1.58


def _num(v):
    """WQB 수확 지표는 문자열로 온다('1.47'). 못 읽으면 None."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _client(user_id: int):
    db = sqlite3.connect(f'file:{os.path.join(ROOT, "data", "hyfe_iqc.db")}?mode=ro', uri=True)
    email = db.execute('SELECT wqb_username FROM users WHERE id=?', (user_id,)).fetchone()[0]
    c = wqb_api.WqbApiClient(email, '')
    if not (c._load_session() and c._session_valid()):
        sys.exit('저장 세션이 만료됐다 — 워커가 재인증한 뒤 다시 돌린다(여기서 인증하지 않음)')
    c.authenticate()            # 위에서 유효 확인 → 저장 세션 재사용 경로만 탄다(POST 없음)
    return c


def _fields(c, region, universe, delay, dataset):
    out, offset = [], 0
    while True:
        j = wds._fetch_page(c, region, universe, delay, offset, dataset)
        res = (j or {}).get('results') or []
        out += res
        offset += len(res)
        if not res or offset >= int((j or {}).get('count') or 0):
            return out


def summarize(rows: list[dict]) -> dict:
    """과제 2 양식의 [데이터셋 정보]·[유망 필드] 재료."""
    low_cov = [r for r in rows if (r.get('coverage') or 0) <= 0.30]
    measured = [r for r in rows if r.get('sharpe') is not None]
    const = [r for r in measured if not r['sharpe'] and not r.get('turnover')]
    seen, dup = set(), []
    for r in measured:
        key = (r['sharpe'], r.get('turnover'), r.get('fitness'))
        if r not in const and key in seen:
            dup.append(r)
        seen.add(key)
    promising = sorted((r for r in measured if r not in dup
                        and abs(r['sharpe'] or 0) >= PROMISING_SHARPE),
                       key=lambda r: -abs(r['sharpe']))
    return {'fields': len(rows), 'measured': len(measured), 'low_coverage': len(low_cov),
            'constant': len(const), 'duplicate': len(dup),
            'promising': [{k: r.get(k) for k in ('field', 'sharpe', 'fitness', 'turnover',
                                                  'margin', 'coverage', 'alpha')}
                          for r in promising]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('region'); ap.add_argument('universe'); ap.add_argument('dataset')
    ap.add_argument('--delay', type=int, default=1)
    ap.add_argument('--template', required=True, help="{f} 자리에 필드가 들어간다")
    ap.add_argument('--neut', default='INDUSTRY')
    ap.add_argument('--decay', type=int, default=0)
    ap.add_argument('--truncation', type=float, default=0.08)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--user-id', type=int, default=2)
    a = ap.parse_args()

    c = _client(a.user_id)
    meta = {f['id']: f for f in _fields(c, a.region, a.universe, a.delay, a.dataset)}
    print(f'{a.dataset}: 필드 {len(meta)}개', flush=True)
    settings = {'region': a.region, 'universe': a.universe, 'delay': a.delay,
                'neutralization': a.neut, 'decay': a.decay, 'truncation': a.truncation}
    specs = []
    for fid, f in meta.items():
        operand = f'vec_avg({fid})' if str(f.get('type')).upper() == 'VECTOR' else fid
        specs.append((fid, a.template.replace('{f}', operand), settings))
    res = {r['tag']: r for r in campaign.run(c, specs, workers=a.workers, deadline_s=1800,
                                              log_fn=lambda m: print(m, flush=True))}
    rows = [{'field': fid, 'type': f.get('type'), 'coverage': f.get('coverage'),
             'alpha': res.get(fid, {}).get('alpha'), 'expr': res.get(fid, {}).get('expr'),
             **{k: _num(res.get(fid, {}).get(k)) for k in ('sharpe', 'fitness', 'turnover', 'margin')}}
            for fid, f in meta.items()]
    path = os.path.join(ROOT, '_workspace', f'scan_{a.region}_{a.dataset}.json')
    with open(path, 'w') as fh:                 # 요약이 죽어도 원자료는 남긴다
        json.dump({'settings': settings, 'template': a.template, 'rows': rows}, fh,
                  ensure_ascii=False, indent=1)
    summary = summarize(rows)
    with open(path, 'w') as fh:
        json.dump({'settings': settings, 'template': a.template, 'summary': summary,
                   'rows': rows}, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'promising'}, ensure_ascii=False))
    for p in summary['promising'][:10]:
        print(p)
    print('→', path)


if __name__ == '__main__':
    main()
