# tests/test_discovery_and_decorrelate.py
# 2026-08-18 사장 지시로 고친 세 가지. 근거는 08-17~18 라이브 실측(알파 300여 건)이다.
import random

from server import genome_models as gm
from server import submit_push as sp
from server import wqb_api


def test_discovery_rejects_zero_usage_fields():
    """발굴이 저사용 오름차순이라, 구간 필터가 없으면 죽은 필드가 웨이브를 다 먹는다."""
    assert sp.SIGNAL_USAGE_MIN >= 30   # 9/23: 사용 46·92 필드가 통과
    assert sp.SIGNAL_USAGE_MAX <= 5000


def test_decorrelate_raises_decay():
    """상관은 decay 로만 실제로 움직였다 — 부모보다 **올리는 쪽**이어야 한다.

    실측: 합성 decay 12→24 로 첫 제출, srisk 조립 24→36 으로 둘째 제출.
    """
    base = gm.BaseGenomeModel(round_num=1)
    parent = base._genome(1, random.Random(0))
    parent = gm.Genome(**{**parent.__dict__, 'decay': 4})
    raised = 0
    for seed in range(40):
        m = gm.BaseGenomeModel(round_num=2)
        m.fail_items = ['PROD_CORRELATION(0.84 vs 0.7)']
        m.parent_metrics = {}
        child = m._mutate(parent, random.Random(seed), directed=True)
        if m._last_directive == 'decorrelate':
            raised += 1
            assert child.decay >= 16, f'decay 가 안 올랐다: {child.decay}'
    assert raised, 'decorrelate 지시가 한 번도 안 뽑혔다 — 매핑을 확인하라'


def test_prod_correlation_maps_to_the_same_axis():
    """PROD 상관도 self 상관과 같은 축으로 가야 어제 통한 경로가 재현된다."""
    from server import mutation_learn
    assert mutation_learn.categorize(['PROD_CORRELATION(0.84 vs 0.7)']) == ['correlation']
    assert mutation_learn.RULE_DIRECTIVE['correlation'] == 'decorrelate'


def test_submit_deadline_outlives_the_wqb_battery():
    """480초로는 08-17 하루에만 10회 넘게 끊겼다 — 판정을 못 받으면 되쏴야 한다."""
    assert wqb_api._SUBMIT_ALPHA_DEADLINE_S >= 900
