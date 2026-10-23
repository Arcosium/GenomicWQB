import random

from server import alpha_ast, alpha_lint, glb_legs


def test_candidates_are_valid_single_dataset_regular_cut_alphas():
    cands = glb_legs.candidates(5, rng=random.Random(7))
    assert len(cands) == 5 and len({c['code'] for c in cands}) == 5
    for c in cands:
        code = c['code']
        assert c['origin'] == 'glb_legs' and c['settings']['region'] == 'GLB'
        assert ', filter=true) + ' in code                  # 다리 add 뒤에 앵커 하나
        fields = alpha_ast.fields_used(code)
        assert 5 <= len(fields) <= 7 and all(f.startswith('qfl_') for f in fields)
        assert code.count('(') <= 64                         # WQB 식당 연산자 한도
        assert alpha_lint.validate_alpha(code) == []
    assert glb_legs.candidates(5, rng=random.Random(7)) == cands   # 라운드 rng 로 재현
