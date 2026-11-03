"""제출 뒤 OS 관측. IS 지표와 방문 횟수는 보존하고 밴딧 선택에 작은 보정을 더한다."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
import math
import time

from . import bandit, db

OS_WEIGHT = 0.02
MIN_AGE_DAYS = 30


def number(value):
    try:
        value = float(value) if value is not None and not isinstance(value, bool) else None
    except (ValueError, TypeError):
        return None
    return value if value is not None and math.isfinite(value) else None


def submitted_ts(value):
    try:
        date = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return date.timestamp() if date.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def os_score(snapshot: dict, now: float):
    """최소 30일 지난 OS만 사용. 결측/빈 OS를 손실로 학습하지 않는다."""
    ts = submitted_ts(snapshot.get('dateSubmitted'))
    metrics = snapshot.get('os') or {}
    sharpe = number(metrics.get('sharpe'))
    returns = number(metrics.get('returns'))
    turnover = number(metrics.get('turnover'))
    if (snapshot.get('stage') != 'OS' or ts is None or now - ts < MIN_AGE_DAYS * 86400
            or sharpe is None or returns is None or turnover is None or turnover <= 0):
        return None
    return max(-1.0, min(1.0, sharpe / 3.0))


def assignment(row, snapshot):
    """실제 설정만 귀속한다. 값이 없으면 임의 arm을 만들지 않는다."""
    settings = snapshot.get('settings') or {}
    try:
        genome = json.loads(row.get('genome') or '{}')
    except (ValueError, TypeError):
        genome = {}
    if not isinstance(genome, dict):
        genome = {}
    values = {k: settings.get(k, row.get(k)) for k in ('universe', 'neutralization', 'decay')}
    if any(v is None or v == '' for v in values.values()):
        return []
    values['decay_bucket'] = bandit.decay_to_bucket(values.pop('decay'))
    values['family'] = genome.get('family')
    values['combine'] = genome.get('combine')
    return bandit.arm_keys_for_assignment(values)


def refresh(user_id: int, client, stop_event, *, limit=200, now=None):
    """저장 세션의 GET만 사용한다. 개별 결측은 기록하고 계속하며, 일시 장애 시 다음 틱에서 재시도한다."""
    now = time.time() if now is None else now
    count = 0
    for row in db.post_submission_candidates(user_id, now - 86400, limit):
        if stop_event.is_set():
            break
        snapshot = client.harvest_post_submission(row['wqb_alpha_id'])
        if snapshot is None:
            break
        db.save_post_submission(user_id, row['id'], row['wqb_alpha_id'], snapshot,
                                assignment(row, snapshot), os_score(snapshot, now), now)
        count += 1
        if stop_event.wait(0.25):
            break
    return count


def bandit_adjustments(user_id: int, *, now=None):
    """알파당 최신 관측 하나만 반영. 표본이 적으면 5개 관측만큼 수축한다."""
    now = time.time() if now is None else now
    scores = defaultdict(list)
    for row in db.post_submission_scores(user_id, now - 7 * 86400):
        for key in json.loads(row['arm_keys']):
            scores[key].append(row['score'])
    return {key: OS_WEIGHT * sum(values) / (len(values) + 5)
            for key, values in scores.items()}
