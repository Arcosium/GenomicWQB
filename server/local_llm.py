"""HYFE_IQC 로컬 LLM seam — google.genai 드롭인 shim (외부 API 미사용).

Gemini(google-genai) 호출을 로컬 SGLang(OpenAI 호환 /v1/chat/completions)로 돌린다.
gemini_strategist.py / auth.py 의 import 두 줄만 이 모듈로 바꾸면:
  - genai.Client(...).models.generate_content(model, contents, config) → 로컬 모델 호출
  - prompt cache(client.caches.create) → 안정된 system·카탈로그 접두부 → SGLang KV 캐시 재사용
  - Google Search grounding(tools=...) / response_schema / ThinkingConfig → 무해한 no-op
모델명(gemini-*)은 무시하고 항상 LOCAL_LLM_MODEL 을 쓴다. JSON 산출은 프롬프트가 이미
"JSON 배열만" 을 강제하고 호출부 _parse_strategies 가 코드펜스/[...] 추출을 처리한다.
"""
import os
import hashlib
import threading
import time

import requests

LOCAL_BASE_URL = os.environ.get('LOCAL_LLM_BASE_URL', 'http://127.0.0.1:11434/v1').rstrip('/')
LOCAL_MODEL = os.environ.get('LOCAL_LLM_MODEL', 'arc-local')
_TIMEOUT = int(os.environ.get('LOCAL_LLM_TIMEOUT', '600'))


class _Resp:
    """google.genai 응답 호환 — 호출부는 resp.text 만 읽는다."""
    def __init__(self, text):
        self.text = text or ''


_CACHE = {}
_CACHE_LOCK = threading.RLock()


def _content_text(contents):
    if isinstance(contents, str): return contents
    if not isinstance(contents, list): return str(contents)
    return '\n\n'.join(str(getattr(part, 'text', '')) for content in contents for part in getattr(content, 'parts', []))


class _Models:
    def generate_content(self, *, model=None, contents='', config=None):
        system = getattr(config, 'system_instruction', None) if config is not None else None
        temperature = getattr(config, 'temperature', 0.9) if config is not None else 0.9
        max_tokens = getattr(config, 'max_output_tokens', 4096) if config is not None else 4096
        messages = []
        name = getattr(config, 'cached_content', None) if config is not None else None
        cached = None
        if name:
            with _CACHE_LOCK:
                cached = _CACHE.get(name)
            if cached is None or cached['expires'] < time.monotonic():
                raise ValueError('cached_content expired; regenerate cache')
            if system and str(system) != cached['system']:
                raise ValueError('cached_content system mismatch')
            system = cached['system']
        if system:
            messages.append({'role': 'system', 'content': str(system)})
        if cached:
            messages.append({'role': 'user', 'content': cached['content']})
        messages.append({'role': 'user', 'content': _content_text(contents)})
        body = {
            'model': LOCAL_MODEL,                 # gemini-* 무시, 로컬 모델 고정
            'messages': messages,
            'temperature': float(temperature) if temperature is not None else 0.9,
            'max_tokens': int(max_tokens) if max_tokens else 4096,
            'stream': False,
            'chat_template_kwargs': {'enable_thinking': False},
            'top_p': 0.8, 'top_k': 20,
        }
        r = requests.post(LOCAL_BASE_URL + '/chat/completions', json=body, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        text = ((data.get('choices') or [{}])[0].get('message') or {}).get('content', '') or ''
        return _Resp(text)


class _Caches:
    def __init__(self, namespace):
        self.namespace = namespace

    def create(self, *, model=None, config=None):
        system = str(getattr(config, 'system_instruction', '') or '')
        content = _content_text(getattr(config, 'contents', []))
        if not content: return None
        name = 'local-prefix-' + hashlib.sha256((self.namespace + '\0' + system + '\0' + content).encode()).hexdigest()
        ttl = str(getattr(config, 'ttl', '3600s')).rstrip('s')
        with _CACHE_LOCK:
            now = time.monotonic()
            for key in list(_CACHE):
                if _CACHE[key]['expires'] < now: del _CACHE[key]
            _CACHE[name] = {'system': system, 'content': content, 'expires': now + min(3600, max(1, int(ttl)))}
        return _Stub(name=name)


class Client:
    def __init__(self, *a, **k):
        namespace = hashlib.sha256(str(k.get('api_key') or '').encode()).hexdigest()
        self.models = _Models()
        self.caches = _Caches(namespace)


# ── google.genai.types 드롭인 no-op 스텁 (모듈 로드시 스키마/툴 구성이 깨지지 않게) ──
class _Type:
    ARRAY = 'ARRAY'
    OBJECT = 'OBJECT'
    STRING = 'STRING'
    NUMBER = 'NUMBER'
    INTEGER = 'INTEGER'
    BOOLEAN = 'BOOLEAN'


class _Stub:
    """모든 키워드 인자를 속성으로 보관하는 범용 no-op (Config/Schema/Tool/Content 등)."""
    def __init__(self, *a, **k):
        self.__dict__.update(k)


class _TypesModule:
    Type = _Type
    GenerateContentConfig = _Stub
    Schema = _Stub
    CreateCachedContentConfig = _Stub
    Content = _Stub
    Part = _Stub
    Tool = _Stub
    GoogleSearch = _Stub
    ThinkingConfig = _Stub


types = _TypesModule()
