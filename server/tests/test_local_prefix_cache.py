import importlib.util
from types import SimpleNamespace as S
import pytest
spec=importlib.util.spec_from_file_location('local_llm','/home/arcosium/projects/GenomicWQB/server/local_llm.py');llm=importlib.util.module_from_spec(spec);spec.loader.exec_module(llm)

def test_catalog_is_preserved_before_changing_round(monkeypatch):
 sent=[]
 class R:
  def raise_for_status(self):pass
  def json(self):return {'choices':[{'message':{'content':'[]'}}]}
 monkeypatch.setattr(llm.requests,'post',lambda *a,**k:sent.append(k['json']) or R())
 client=llm.Client(api_key='test');cached=client.caches.create(config=S(system_instruction='규칙',contents='고정 카탈로그',ttl='3600s'))
 for rnd in (1,2):
  llm.Client(api_key='test').models.generate_content(contents=f'라운드 {rnd}',config=S(cached_content=cached.name))
 assert sent[0]['messages'][:2]==sent[1]['messages'][:2]==[{'role':'system','content':'규칙'},{'role':'user','content':'고정 카탈로그'}]
 assert sent[0]['messages'][-1]!=sent[1]['messages'][-1]

def test_expired_prefix_never_drops_catalog(monkeypatch):
 client=llm.Client(api_key='test');cached=client.caches.create(config=S(system_instruction='규칙',contents='내용',ttl='1s'))
 llm._CACHE[cached.name]['expires']=0
 with pytest.raises(ValueError,match='expired'):client.models.generate_content(contents='new',config=S(cached_content=cached.name))


def test_generated_candidates_disable_reasoning(monkeypatch):
 sent=[]
 class R:
  def raise_for_status(self):pass
  def json(self):return {'choices':[{'message':{'content':'[]'}}]}
 monkeypatch.setattr(llm.requests,'post',lambda *a,**k:sent.append(k['json']) or R())
 llm.Client().models.generate_content(contents='후보 생성',config=S(max_output_tokens=128))
 assert sent[0]['chat_template_kwargs']=={'enable_thinking':False}
 assert sent[0]['max_tokens']==128
