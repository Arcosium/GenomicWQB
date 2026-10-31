"""superalpha — OS 알파 풀 위의 슈퍼알파 리서치 경로 (⑤, WQB AAF SuperAlpha 이식).

AAF 레시피 그대로: 필터(롱숏비·turnover·datafield 수·OS 시작일) → 의사난수
선택(turnover/long_count 를 시드로 쓰는 ACE 트릭 — 특정 위험으로의 쏠림 없이
복원추출) → combo 그리드(alpha_stats × {rank, ts_rank} × 부호) →
type=SUPER 시뮬 → **FAIL 0 이면 바로 제출**(런당 1건 성공까지).

2026-10-31 실측: RC 계정에 super simulation 권한이 열려 있다(7/27 엔 NOT_PERMISSIONED).
슈퍼알파는 제출 한도가 따로다(체크 SUPER_SUBMISSION — REGULAR·PP 월 한도와 별개).
그래서 일반 제출이 PP 월 한도에 막힌 날에도 낼 수 있다. 워커 드레인 티커가 하루 1런을
돌리고(Worker._maybe_superalpha), /api/superalpha/run 으로 수동 1런도 된다.
게이트: IQC_SUPERALPHA (기본 1, 0 이면 끔) + RC 계정.
표현식 생성부는 순수 함수 (run() 만 네트워크/DB).

ponytail: 선택 셋 자체의 PnL 상관검사(원본 노트북의 max_corr<0.5 드롭)는 뺐다 —
알파당 PnL GET 이 필요해 쿼터가 아깝다. 슈퍼알파 시뮬 결과가 나쁘면 그게 곧
같은 신호다. 필요해지면 get_alpha_pnl 상관 게이트를 여기 run() 에 추가할 것.
"""
from __future__ import annotations

import logging
import random
import threading
import time

from . import criteria as _criteria
from . import db as _db

LOG = logging.getLogger('genomicwqb.superalpha')

SELECTION_LIMIT = 10
SELECTION_HANDLING = 'NON_NAN'
# 429 = 동시 시뮬 슬롯이 찼다(라운드 시뮬과 같은 슬롯을 쓴다). 버리지 않고 이만큼 기다린다.
RATE_LIMIT_WAIT_S = 1200

# combo 그리드 축 (AAF: 이 셋 밖 연산자는 효율이 나빴다고 명시).
# ts_ir 은 2026-10-31 실측 'inaccessible or unknown operator' — 시뮬 쿼터만 태워서 뺐다.
COMBO_OPS = ('rank', 'ts_rank')
COMBO_STATS = ('hold_value', 'short_count/stats.long_count', 'pnl', 'returns',
               'turnover')


def selection_expression(seed_plus: int, *,
                         datafield_count_max: int = 8,
                         longshort_bal: float = 0.1,
                         os_start_date: str = '2023-01-01') -> str:
    """필터 조건 × 의사난수 정렬키 (ACE 노트북 형식 그대로).

    sqrt(turnover+d)*10000 의 소수부는 d 만 바뀌어도 사실상 난수 — 알파 스탯을
    시드로 쓰는 복원추출이라 특정 위험 노출로 쏠리지 않는다(AAF Sort-alphas).
    """
    turnover_max = _criteria.SUPERALPHA_TURNOVER_MAX
    conditions = '*'.join([
        f'(datafield_count < {int(datafield_count_max)})',
        f'(abs(short_count/long_count-1) < {longshort_bal})',
        f'(turnover < {turnover_max})',
        f'(turnover > {_criteria.SUPERALPHA_TURNOVER_MIN})',
        f'(os_start_date > "{os_start_date}")',
    ])
    return (f'd = {int(seed_plus)};\n'
            f'{conditions}*\n'
            f'(sqrt(turnover+d)*10000-round(sqrt(turnover+d)*10000)+\n'
            f'sqrt(long_count+d)*10000-round(sqrt(long_count+d)*10000))')


# 설명 100자 미만이면 제출 403 (COMBO/SELECTION_DESCRIPTION_LENGTH, 2026-10-31 실측).
SELECTION_DESCRIPTION = (
    'Idea: pick up to 10 of my own submitted alphas that are cheap to combine - fewer than 8 '
    'datafields, balanced long/short books, turnover between 2% and 40%, OS start after '
    '2023-01-01. Rationale: a pseudo-random sort key seeded by each alpha\'s own stats samples '
    'the pool without tilting toward one risk factor.')


def combo_description(combo: str) -> str:
    body = str(combo).split(';', 1)[-1].strip()
    return (f'Idea: weight each selected alpha by {body} computed from its own track record. '
            'Rationale: components with stronger recent statistics get more weight, while the '
            'rank transform keeps weights bounded so no single component dominates.')


def combo_grid() -> list[str]:
    """alpha_stats × 연산자 (+ -drawdown 축) — AAF Combine-alphas 그리드."""
    out = []
    for stat in COMBO_STATS:
        for op in COMBO_OPS:
            win = ',126' if op.startswith('ts') else ''
            out.append(f'stats = generate_stats(alpha);\n{op}(stats.{stat}{win})')
    for op in COMBO_OPS:
        win = ',126' if op.startswith('ts') else ''
        out.append(f'stats = generate_stats(alpha);\n{op}(-stats.drawdown{win})')
    return out


def default_settings() -> dict:
    """SUPER 시뮬 settings 완성본 (ACE generate_super_alpha 기본값, decay 만 10).

    decay 는 ACE 기본 0 이면 회전율 0.56~0.67 로 슈퍼알파 상한 0.4(HIGH_TURNOVER)를
    4/4 넘었다. 같은 selection·rank(stats.turnover) 실측(2026-10-31): decay 5 → TO 0.31·
    S 2.52, decay 10 → TO 0.21·S 2.25, 둘 다 나머지 체크 PASS. 회전율이 더 높은 combo 도
    상한 안에 들도록 10.
    """
    return {
        'nanHandling': 'OFF', 'instrumentType': 'EQUITY', 'delay': 1,
        'universe': 'TOP3000', 'truncation': 0.08, 'unitHandling': 'VERIFY',
        'pasteurization': 'ON', 'region': 'USA', 'language': 'FASTEXPR',
        'decay': 10, 'neutralization': 'INDUSTRY', 'visualization': False,
        'selectionHandling': SELECTION_HANDLING,
        'selectionLimit': SELECTION_LIMIT,
    }


def build_candidates(seed_plus: int, *, n: int = 6, rng=None) -> list[dict]:
    """selection 1개 × combo 그리드에서 n 개 표본 → [{selection, combo, settings}]."""
    sel = selection_expression(seed_plus)
    combos = combo_grid()
    if rng is not None and n < len(combos):
        combos = rng.sample(combos, n)
    else:
        combos = combos[:n]
    st = default_settings()
    return [{'selection': sel, 'combo': c, 'settings': dict(st)} for c in combos]


def run(user_id: int, username: str, password: str, *,
        seed_plus: int = 10, n: int = 6, rng=None,
        stop_event=None, log_fn=None) -> int | None:
    """슈퍼알파 리서치 1런 — 백그라운드 스레드에서 호출된다. → run_id.

    후보를 순차 시뮬한다(ACE 권장 동시 2 이지만 일일 쿼터 공유라 순차가 안전).
    FAIL 0 인 첫 결과는 곧장 제출한다 — 거절은 한도를 쓰지 않고, 상관(SELF·PROD)은
    제출 때에만 판정된다. 결과/제출/실패는 superalpha_runs 에 영속화. 예외는 error 로
    기록하고 삼킨다.

    성공 1건에서 제출을 멈춘다 — SUPER_SUBMISSION 한도가 하루 1건이다(2026-10-31 실측
    limit 1). 제출 결과는 submit_attempts 에 넣지 않는다(그 테이블의 성공 수가 일반 제출
    하루 4건 예산이라 슈퍼가 그 칸을 먹는다).
    """
    log = log_fn or (lambda m: LOG.info('%s', m))
    cands = build_candidates(seed_plus, n=n, rng=rng)
    run_id = _db.superalpha_start(user_id, seed_plus, cands[0]['selection'])
    results: list[dict] = []
    submitted = False
    try:
        from . import wqb_api as _wqb_api
        client = _wqb_api.WqbApiClient(username, password)
        if not client.authenticate():
            _db.superalpha_finish(run_id, 'error', results, 'WQB 인증 실패')
            return run_id
        for c in cands:
            if stop_event is not None and stop_event.is_set():
                break
            t0 = time.time()
            while True:
                loc = client.submit_super_simulation(c['selection'], c['combo'],
                                                     c['settings'])
                if loc != 'RATE_LIMITED' or time.time() - t0 > RATE_LIMIT_WAIT_S:
                    break
                if stop_event is not None and stop_event.wait(30):
                    break                   # 일시정지 — 기다리지 않는다
                if stop_event is None:
                    time.sleep(30)
            if loc == 'NOT_PERMISSIONED':
                # 계정 권한 문제 — 남은 후보를 돌려봐야 전부 같은 400 이다. 즉시 끝낸다.
                _db.superalpha_finish(
                    run_id, 'error', results,
                    'WQB 계정에 super simulation 권한이 없습니다. CONSULTANT 권한만으로는 '
                    '안 되고 별도 권한이 필요합니다 — WorldQuant 에 요청하세요.')
                return run_id
            if loc in (None, 'RATE_LIMITED'):
                results.append({'combo': c['combo'], 'error': str(loc or 'submit 실패')})
                continue
            done = client.poll(loc, stop_event=stop_event)
            alpha_id = done.get('alpha')
            entry: dict = {'combo': c['combo'], 'alpha_id': alpha_id,
                           'status': done.get('status')}
            if alpha_id:
                try:
                    h = client.harvest_alpha(alpha_id) or {}
                    entry['metrics'] = h.get('metrics') or {}
                    entry['is_status'] = h.get('is_status') or {}
                except Exception as e:
                    entry['error'] = f'harvest: {e}'
                st = entry.get('is_status') or {}
                fails = [f.get('name') if isinstance(f, dict) else f
                         for f in st.get('fail') or []]
                m = entry.get('metrics') or {}
                log(f'  🧬 슈퍼알파 {alpha_id} — S {m.get("sharpe")} · F {m.get("fitness")}'
                    f' · TO {m.get("turnover")} · FAIL {fails or 0}')
                # PASS 가 하나도 없으면(체크 미수신·ERROR 뿐) 판정 불가 — 내지 않는다.
                # harvest 는 체크가 없어도 빈 목록 dict 를 주므로 `st` 참거짓으론 못 거른다.
                # 제출 방식이 '목록에 추가'(list)면 자동 제출하지 않는다 — 큐 드레인과 같은 규칙.
                if (st.get('pass') and not fails and not submitted
                        and _db.get_submit_mode(user_id) != 'list'):
                    client.set_super_description(alpha_id, SELECTION_DESCRIPTION,
                                                 combo_description(c['combo']))
                    ok, msg = client.submit_alpha(alpha_id, stop_event=stop_event,
                                                  deadline_s=600)
                    entry['submit'] = {'ok': bool(ok), 'status': str(msg or '')[:300]}
                    submitted = bool(ok)
                    log(f'  🚀 슈퍼알파 제출 성공 — {alpha_id}' if ok
                        else f'  ⛔ 슈퍼알파 {alpha_id}: {str(msg or "")[:160]}')
            else:
                entry['error'] = str(done.get('message') or '')[:200]
            results.append(entry)
        _db.superalpha_finish(run_id, 'done', results)
    except Exception as e:
        LOG.exception('superalpha run 실패')
        _db.superalpha_finish(run_id, 'error', results, str(e))
    return run_id


def start_background(user_id: int, username: str, password: str, *,
                     seed_plus: int = 10, n: int = 6, log_fn=None,
                     stop_event=None) -> None:
    # rng 없이 돌리면 build_candidates 가 매번 그리드 앞 n 개만 고른다 — 시드로 섞는다.
    # stop_event = 워커 일시정지 — 남은 시뮬·제출을 멈춘다.
    t = threading.Thread(
        target=run, args=(user_id, username, password),
        kwargs={'seed_plus': seed_plus, 'n': n, 'log_fn': log_fn,
                'rng': random.Random(seed_plus), 'stop_event': stop_event},
        daemon=True, name=f'superalpha-{user_id}')
    t.start()
