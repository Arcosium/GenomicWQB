# 하우스 RC 이메일은 소스에 두지 않는다 — env → data/house_rc_username.txt → '' 순.
from server import wqb_data_service as wds


def test_env_wins(monkeypatch, tmp_path):
    monkeypatch.setenv('HYFE_HOUSE_RC_USERNAME', 'a@b.example')
    assert wds._house_rc_username(str(tmp_path / 'none.txt')) == 'a@b.example'


def test_file_fallback_then_empty(monkeypatch, tmp_path):
    monkeypatch.delenv('HYFE_HOUSE_RC_USERNAME', raising=False)
    f = tmp_path / 'u.txt'
    assert wds._house_rc_username(str(f)) == ''           # 파일도 없으면 빈 값 → 폴백
    f.write_text('house@rc.example\n')
    assert wds._house_rc_username(str(f)) == 'house@rc.example'


def test_source_has_no_hardcoded_mailbox():
    import re
    src = open(wds.__file__, encoding='utf-8').read()
    assert not re.search(r'[\w.+-]+@(?:gmail|naver|daum|hanmail)\.com', src)
