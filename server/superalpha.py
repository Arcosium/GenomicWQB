"""superalpha — 제출 알파 풀 위의 슈퍼알파 리서치 경로 (⑤).

GLB 풀에서 데이터 카테고리별 선택(self_correlation 낮은 순) × combo_a →
type=SUPER 시뮬 → **FAIL 0 이면 바로 제출**(런당 1건 성공까지). 제출 SELF 검사가
앞서 낸 슈퍼알파와 겹치는 후보를 거르므로 날마다 다른 카테고리가 나간다.
(2026-11-11 고급 스프린트 과제 2 — 예전 AAF 의사난수·USA 레시피는 전부 PROD 거절이라 교체.)

2026-10-31 실측: RC 계정에 super simulation 권한이 열려 있다(7/27 엔 NOT_PERMISSIONED).
슈퍼알파는 제출 한도가 따로다(체크 SUPER_SUBMISSION — REGULAR·PP 월 한도와 별개).
그래서 일반 제출이 PP 월 한도에 막힌 날에도 낼 수 있다. 워커 드레인 티커가 하루 1런을
돌리고(Worker._maybe_superalpha), /api/superalpha/run 으로 수동 1런도 된다.
게이트: IQC_SUPERALPHA (기본 1, 0 이면 끔) + RC 계정.
표현식 생성부는 순수 함수 (run() 만 네트워크/DB).

ponytail: 선택 셋 안의 PnL 상관검사는 뺐다 — 선택식의 self_correlation 정렬과 제출 SELF
검사로 충분했다(2026-11-11). 필요해지면 get_alpha_pnl 상관 게이트를 run() 에 추가할 것.
"""
from __future__ import annotations

import logging
import random
import threading
import time

from . import db as _db

LOG = logging.getLogger('genomicwqb.superalpha')

SELECTION_LIMIT = 15
SELECTION_HANDLING = 'POSITIVE'
# 429 = 동시 시뮬 슬롯이 찼다(라운드 시뮬과 같은 슬롯을 쓴다). 버리지 않고 이만큼 기다린다.
RATE_LIMIT_WAIT_S = 1200

# GLB 풀(제출 200개)을 데이터 카테고리 등으로 나눠 묶음마다 슈퍼알파 하나 — 재료가 덜 겹쳐
# 서로 SELF 도 낮다. 2026-11-11 실측(MINVOL1M·COUNTRY·d6·combo_a), 제출 없는 상관 조회:
#   pv S4.23·PROD 0.63·SELF 0.35 / multi S4.58·0.61·0.41 / model S2.44·0.62·0.47 /
#   other S4.14·0.69·0.46 — 통과. fundamental 을 self 순으로 고르면 PROD 0.73, risk 는 0.80,
#   analyst 는 IS_LADDER 1.13, sentiment·news 는 재료 10개 미만으로 실패.
# 예전 USA 의사난수 선택(풀 36개)은 10/31~11/10 66건 전부 PROD 0.82~0.91 로 거절됐다.
# 하루 1런이 시드(날짜)만큼 순서를 돌려 시작점을 바꾼다.
SELECTIONS = {
    'pv': 'in(datacategories, "pv") * (turnover < 0.3) * (2 - self_correlation)',
    'multi': '(datacategory_count >= 2) * (turnover < 0.3) * (2 - self_correlation)',
    'model': 'in(datacategories, "model") * (turnover < 0.3) * (2 - self_correlation)',
    'other': 'in(datacategories, "other") * (turnover < 0.3) * (2 - self_correlation)',
    'fundamental': 'in(datacategories, "fundamental") * (turnover < 0.3) * (2 - prod_correlation)',
}
CATEGORIES = tuple(SELECTIONS)
COMBO = 'combo_a(alpha)'


def selection_expression(category: str) -> str:
    """그 묶음의 GLB 알파를 기존 알파(또는 프로덕션)와 덜 겹치는 순으로."""
    return SELECTIONS[category]


# 설명 100자 미만이면 제출 403 (COMBO/SELECTION_DESCRIPTION_LENGTH, 2026-10-31 실측).
def selection_description(category: str) -> str:
    group = ('that mix two or more data categories' if category == 'multi'
             else f'that use {category} data')
    return (f'Idea: combine my submitted GLB alphas {group}, keeping only those with turnover '
            'below 30%. Rationale: one group per SuperAlpha keeps its components economically '
            'related while separate SuperAlphas stay uncorrelated; components that overlap '
            'least with existing alphas are ranked first.')


COMBO_DESCRIPTION = (
    'Idea: combo_a weights each selected alpha by its recent return relative to its volatility '
    'over the last year. Rationale: components with steadier recent performance get more '
    'weight, and no fitted parameters are added on top of the selected alphas.')


def default_settings() -> dict:
    """SUPER 시뮬 settings 완성본 — Learn 'Global SuperAlphas' 권장(COUNTRY 중립화·MINVOL1M).

    decay 0 이면 회전율이 슈퍼알파 상한 0.4 를 넘는다(2026-10-31 USA 실측). GLB d6 은
    회전율 0.07~0.18(2026-11-11).
    """
    return {
        'nanHandling': 'OFF', 'instrumentType': 'EQUITY', 'delay': 1,
        'universe': 'MINVOL1M', 'truncation': 0.08, 'unitHandling': 'VERIFY',
        'pasteurization': 'ON', 'region': 'GLB', 'language': 'FASTEXPR',
        'decay': 6, 'neutralization': 'COUNTRY', 'visualization': False,
        'maxTrade': 'OFF', 'componentActivation': 'IS',
        'selectionHandling': SELECTION_HANDLING,
        'selectionLimit': SELECTION_LIMIT,
    }


def build_candidates(seed_plus: int, *, n: int = 6, rng=None) -> list[dict]:
    """카테고리를 seed 만큼 돌려 n 개 → [{selection, combo, settings, category}]. rng 는 안 쓴다."""
    k = int(seed_plus) % len(CATEGORIES)
    cats = (CATEGORIES[k:] + CATEGORIES[:k])[:n]
    st = default_settings()
    return [{'selection': selection_expression(c), 'combo': COMBO, 'settings': dict(st),
             'category': c} for c in cats]


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
                    client.set_super_description(alpha_id,
                                                 selection_description(c['category']),
                                                 COMBO_DESCRIPTION)
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
