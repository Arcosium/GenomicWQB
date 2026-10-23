"""glb_legs — GLB quant_factor_lib 다리 조합 레이어 (2026-10-23 사장 지시).

10/21~22 중급 스프린트에서 손으로 찾은 방식을 그대로 라운드에 끼운다.
그때 GLB 시뮬 242건 중 61건이 IS 체크를 전부 통과했고 셋이 일반 관문으로 제출됐다
(rKdRvgOJ·RRAEamEj·vRO15kK3, S 1.87~2.69·PROD 0.55~0.60).

  · 다리 = w*(group_rank(ts_backfill(vec_avg(F),60),country) - 0.5) — 국가 안 순위를 0 중심으로.
  · 다리끼리는 add(..., filter=true) — 커버리지가 다른 필드를 + 로 더하면 NaN 교집합만 남아 무너진다.
  · 앵커 1개를 + 로 붙인다 — 커버리지 100% cassie 다리가 IS 기간을 정하고 PnL 을 받친다.
  · 설정 GLB·MINVOL1M·COUNTRY·d10·tr0.05. 한 데이터셋이라 Single Data Set 태그가 붙는다.

필드 다섯 개 이상이라 PP 자격(연산자≤8·필드≤3)은 없다 — 일반 관문(S 1.58·fitness 1) 겨냥이다.
그래서 worker 는 이 후보를 PP 복잡도 관문과 filter= 제거 repair 에서 뺀다.
"""
from __future__ import annotations

import random as _random

IDX_BASE = 71                      # 재조합 31+ / 개선 41+ / HT구제 51+ / 사냥 사다리 61+ 다음

#: (부호, 필드). 부호·선정은 스프린트 단일 필드 스캔(S20≥0.5, cassie 제외)과
#: 성공작 셋에 쓰인 다리 넷(payble_days·finished_goods·sg_a_vol·claim)에서 왔다.
#: linked_momentum_6m 은 PROD 를 올려 뺐다.
LEGS = (
    (-1, 'gamef_cube_game_non_production_asset'), (1, 'gamef_cube_qes_gamef_cscore'),
    (1, 'patent_innovation_foreign_priority'), (1, 'patent_innovation_intensity'),
    (1, 'patent_innovation_unbiased_inward_citation_efficiency'),
    (1, 'patent_innovation_unbiased_outward_citation'), (1, 'gamef_cube_game_pension_ev'),
    (1, 'patent_self_innovation_intensity'), (-1, 'gamef_cube_game_other_asset_gr'),
    (-1, 'gamef_cube_game_deferred_tax_liab_chg'), (1, 'patent_innovation_pagerank'),
    (1, 'gamef_cube_game_pension_pbo'), (1, 'patent_innovation_originality'),
    (1, 'nemo_cube_bsmearningsrevision'), (1, 'patent_innovation_linked_momentum_1m'),
    (1, 'gamef_cube_qes_gamef_mohanram'), (1, 'nemo_cube_anmearningsrevision'),
    (-1, 'gamef_cube_game_equity_dilution'), (1, 'patent_innovation_diversity'),
    (-1, 'nlp_calltranscript_global_qa_ceo_read_fog'),
    (1, 'nlp_calltranscript_global_qa_sell_lm2_cum_pos'), (1, 'nemo_cube_ebmearningsrevision'),
    (-1, 'gamef_cube_game_inventory_chg'), (1, 'gamef_cube_game_pension_expense'),
    (-1, 'gamef_cube_qes_gamef_earnigs_timeliness'),
    (1, 'nlp_calltranscript_global_pre_ceo_lm2_cum_pos'),
    (1, 'patent_innovation_unbiased_inward_citation'), (1, 'corpexec_cube_key_executive_ice'),
    (1, 'nlp_calltranscript_global_qa_ceo_vader_pos'), (1, 'patent_pia'),
    (1, 'gamef_cube_game_deferred_sal_chg'), (1, 'nemo_cube_ebmearningssurprise'),
    (1, 'nlp_calltranscript_global_qa_ceo_lm2_cum_pos'), (1, 'gamef_cube_qes_gamef_piotroski'),
    (1, 'gamef_cube_game_cash_flow_coverage'), (1, 'corpexec_cube_cfo_tenure'),
    (-1, 'gamef_cube_game_payble_days'), (-1, 'gamef_cube_game_finished_goods'),
    (-1, 'gamef_cube_game_sg_a_vol'), (1, 'patent_innovation_claim'),
)

#: (부호, 필드, 뽑기 가중). 스프린트 통과율: 단기부채 증가 11/25 · altman 3/9 · merton 2/6 ·
#: 1년 부도거리 1/5. altman 은 PnL 을 지배해(Va≈altman 0.74) 서로 self 를 올리니 가중을 낮췄다.
ANCHORS = (
    (1, 'cassie_qes_alter_growth_shortterm_debt', 5),
    (-1, 'cassie_model_altman_zscore', 2),
    (-1, 'cassie_model_merton', 2),
    (-1, 'cassie_qes_alter_distance_insolvency_1y', 1),
)

SETTINGS = {'region': 'GLB', 'universe': 'MINVOL1M', 'neutralization': 'COUNTRY',
            'decay': '10', 'truncation': '0.05'}


def _leg(w: int, field: str) -> str:
    return f'{w}*(group_rank(ts_backfill(vec_avg(qfl_{field}), 60), country) - 0.5)'


def candidates(n: int = 2, rng=None) -> list[dict]:
    """워커 strategies 항목 모양의 후보 n 개 — {idx, code, desc, settings, origin}."""
    rng = rng or _random
    out, made = [], set()
    for _ in range(n * 6):
        if len(out) >= n:
            break
        legs = rng.sample(LEGS, rng.randint(4, 6))
        ws = [s * rng.choice((1, 2)) for s, _f in legs]
        a_sign, a_field, _ = rng.choices(ANCHORS, weights=[a[2] for a in ANCHORS])[0]
        code = ('add(' + ', '.join(_leg(w, f) for w, (_s, f) in zip(ws, legs))
                + ', filter=true) + ' + _leg(a_sign, a_field))
        if code in made:
            continue
        made.add(code)
        out.append({
            'idx': IDX_BASE + len(out),
            'code': code,
            'desc': f'🌐 GLB qfl 다리 {len(legs)}개 + 앵커 {a_field.split("_", 1)[1]}',
            'settings': dict(SETTINGS),
            'origin': 'glb_legs',
        })
    return out
