# v1.2 ① AMT 준비 (GitHub Actions 의 amt-prep 이 불러요)
#  · Anticipatory Music Transformer small (stanford-crfm/music-small-800k · 코드 Apache-2.0 · 가중치 라이선스는 모델 카드에서 확인해 찍어요)
#  · torch.onnx 로 직접 ONNX(KV 캐시, 길이 축 동적) → int8 동적 양자화 → 45MB 조각으로 (GitHub 파일 100MB · jsDelivr 50MB 한도)
#  · 기준값: 정해진 토큰 열의 마지막 로짓 상위 10 (파이토치 fp32 · ORT int8) · 위치별 1위 일치율 · 공식 generate 로 4초 채우기가 도는지
import os, sys, json, hashlib, subprocess, numpy as np, torch
OUT = os.path.abspath('models'); os.makedirs(OUT, exist_ok=True)
MID = 'stanford-crfm/music-small-800k'
from huggingface_hub import hf_hub_download, model_info
try:
    info = model_info(MID); print('모델 카드 라이선스:', getattr(info, 'card_data', None) and info.card_data.get('license'), '· 태그', [t for t in info.tags if 'license' in t])
    rd = open(hf_hub_download(MID, 'README.md')).read(); print('README 앞:', rd[:600].replace('\n', ' | '))
except Exception as e:
    print('모델 카드 확인 실패:', e)
# optimum 없이 직접 내보내기 (GPT-2 · past_key_values 를 평평한 입출력으로, 길이 축 동적) — eager 어텐션: sdpa 는 추적 때 is_causal · 마스크 분기가 굳어요
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
base = AutoModelForCausalLM.from_pretrained(MID, attn_implementation='eager').eval(); cfg = base.config; NL, NH, HD = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head
class Wrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask, position_ids, *past):
        pkv = DynamicCache.from_legacy_cache(tuple((past[2 * i], past[2 * i + 1]) for i in range(NL)))
        o = self.m(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=pkv, use_cache=True)
        pk = o.past_key_values.to_legacy_cache() if hasattr(o.past_key_values, 'to_legacy_cache') else o.past_key_values
        return (o.logits, *[t for kv in pk for t in kv])
w = Wrap(base); P = 3; n = 5
dummy = (torch.randint(0, 100, (1, n)), torch.ones(1, P + n, dtype=torch.long), torch.arange(P, P + n)[None], *[torch.randn(1, NH, P, HD) for _ in range(2 * NL)])
inames = ['input_ids', 'attention_mask', 'position_ids'] + [f'past_key_values.{i}.{k}' for i in range(NL) for k in ('key', 'value')]
onames = ['logits'] + [f'present.{i}.{k}' for i in range(NL) for k in ('key', 'value')]
dyn = {'input_ids': {1: 'seq'}, 'attention_mask': {1: 'tot'}, 'position_ids': {1: 'seq'}, 'logits': {1: 'seq'}}
for nm in inames[3:]: dyn[nm] = {2: 'past'}
for nm in onames[1:]: dyn[nm] = {2: 'tot'}
os.makedirs('onnx', exist_ok=True); fp = 'onnx/model.onnx'
with torch.no_grad():
    torch.onnx.export(w, dummy, fp, opset_version=17, input_names=inames, output_names=onames, dynamic_axes=dyn, do_constant_folding=True, dynamo=False)
print('fp32', os.path.getsize(fp) // 1048576, 'MB (외부 파일 제외)')
import onnxruntime as ort
def onnx_check(path, tag, toks_):
    # 진짜 anticipation 토큰 열(앞 60개)로 비교해요 — 시간 토큰만 잇단 가짜 열(100~159)은 분포 밖이라 로짓이 수백까지 튀어 fp32 반올림만으로도 순위가 뒤집혀요
    toks_ = list(toks_[:60]); se = ort.InferenceSession(path); full = base(torch.tensor([toks_])).logits[0].detach().numpy()
    f = {'input_ids': np.array([toks_], dtype=np.int64), 'attention_mask': np.ones((1, 60), dtype=np.int64), 'position_ids': np.arange(60, dtype=np.int64)[None]}
    for nm in inames[3:]: f[nm] = np.zeros((1, NH, 0, HD), dtype=np.float32)
    o = se.run(None, f); d_full = float(np.abs(o[0][0] - full).max())
    f2 = {'input_ids': np.array([toks_[:40]], dtype=np.int64), 'attention_mask': np.ones((1, 40), dtype=np.int64), 'position_ids': np.arange(40, dtype=np.int64)[None]}
    for nm in inames[3:]: f2[nm] = np.zeros((1, NH, 0, HD), dtype=np.float32)
    o2 = se.run(None, f2); f3 = {'input_ids': np.array([toks_[40:]], dtype=np.int64), 'attention_mask': np.ones((1, 60), dtype=np.int64), 'position_ids': np.arange(40, 60, dtype=np.int64)[None]}
    for k, nm in enumerate(inames[3:]): f3[nm] = o2[1 + k]
    o3 = se.run(None, f3); d_inc = float(np.abs(o3[0][0] - full[40:]).max())
    print(tag, 'ONNX vs 파이토치 최대 차: 처음부터', round(d_full, 4), '· KV 캐시 이어서', round(d_inc, 4), '· 1위 일치', round(float(np.mean(np.argmax(o[0][0], -1) == np.argmax(full, -1))), 4), flush=True)
    # 진단: 분포(KL) · 상위 5 · 최대차가 어디서(위치 · 토큰 · 값) — 학습 때 거의 안 나온 토큰(로짓 수천)의 반올림 차인지, 진짜 계산 차인지
    def lsm(a): a = a - a.max(-1, keepdims=True); return a - np.log(np.exp(a).sum(-1, keepdims=True))
    pt, po = lsm(full), lsm(o[0][0]); kl = np.sum(np.exp(pt) * (pt - po), -1); d = np.abs(o[0][0] - full); i, j = np.unravel_index(np.argmax(d), d.shape)
    t5 = np.mean([len(set(np.argsort(-full[k])[:5]) & set(np.argsort(-o[0][0][k])[:5])) / 5 for k in range(len(full))])
    print('   KL 평균', round(float(kl.mean()), 5), '최대', round(float(kl.max()), 5), '(위치', int(np.argmax(kl)), ') · 상위5', round(float(t5), 4), '· 최대차 위치', int(i), '토큰', int(j), '파이토치', round(float(full[i, j]), 2), 'ONNX', round(float(o[0][0][i, j]), 2), '· 로짓 범위', round(float(full.min()), 1), '~', round(float(full.max()), 1), '· 위치별 최대차', [round(float(x), 2) for x in d.max(-1)[:12]], flush=True)
    return d_full, d_inc
from onnxruntime.quantization import quantize_dynamic, QuantType
q = f'{OUT}/amt-small-q.onnx'
VARIANTS = {'pc-all': dict(per_channel=True, op_types_to_quantize=['MatMul', 'Gemm', 'Gather']), 'pt-all': dict(per_channel=False, op_types_to_quantize=['MatMul', 'Gemm', 'Gather']), 'pc-mm': dict(per_channel=True, op_types_to_quantize=['MatMul', 'Gemm'])}
for k, kw in VARIANTS.items():
    quantize_dynamic(fp, f'onnx/v-{k}.onnx', weight_type=QuantType.QInt8, **kw); print('판', k, os.path.getsize(f'onnx/v-{k}.onnx') // 1048576, 'MB', flush=True)
import onnxruntime as ort
s = ort.InferenceSession(f'onnx/v-pc-all.onnx'); print('입력', [i.name for i in s.get_inputs()][:6], '…', len(s.get_inputs()), '· 출력', [o.name for o in s.get_outputs()][:3])
# ---- 기준 토큰 열: C 장조 멜로디(피아노 0) + 베이스(33) 8초, anticipation 형식 ----
sys.path.insert(0, os.path.abspath('anticipation'))
from anticipation import ops
from anticipation.vocab import *
from anticipation.sample import generate
ev = []
for i, p in enumerate([60, 62, 64, 65, 67, 65, 64, 62, 60, 64, 67, 72, 67, 64, 62, 60]):
    t = i * 50; ev += [TIME_OFFSET + t, DUR_OFFSET + 45, NOTE_OFFSET + 0 * 128 + p]
for i, p in enumerate([36, 41, 43, 36]):
    t = i * 200; ev += [TIME_OFFSET + t, DUR_OFFSET + 190, NOTE_OFFSET + 33 * 128 + p]
ev = ops.sort(ev); toks = [AUTOREGRESS] + ops.pad(ev)
m = base
with torch.no_grad(): lt = m(torch.tensor([toks])).logits[0].numpy()
with torch.no_grad(): ref_fp32 = onnx_check(fp, 'fp32', toks); onnx_check(fp, 'fp32 (가짜 열 100~159)', list(range(100, 160)))
names = [i.name for i in s.get_inputs()]; L = len(toks); feeds = {'input_ids': np.array([toks], dtype=np.int64)}
if 'attention_mask' in names: feeds['attention_mask'] = np.ones((1, L), dtype=np.int64)
if 'position_ids' in names: feeds['position_ids'] = np.arange(L, dtype=np.int64)[None]
nl, nh, hd = NL, NH, HD
for n in names:
    if n.startswith('past_key_values'): feeds[n] = np.zeros((1, nh, 0, hd), dtype=np.float32)
if 'use_cache_branch' in names: feeds['use_cache_branch'] = np.array([False])
def judge(path):
    ss = ort.InferenceSession(path); lo_ = ss.run(['logits'], feeds)[0][0]
    def lsm(a): a = a - a.max(-1, keepdims=True); return a - np.log(np.exp(a).sum(-1, keepdims=True))
    pt, po = lsm(lt), lsm(lo_); kl = float(np.mean(np.sum(np.exp(pt) * (pt - po), -1)))
    t5 = float(np.mean([len(set(np.argsort(-lt[i])[:5]) & set(np.argsort(-lo_[i])[:5])) / 5 for i in range(len(lt))]))
    return lo_, {'agree': float(np.mean(np.argmax(lo_, -1) == np.argmax(lt, -1))), 'kl': kl, 'top5': t5, 'mb': os.path.getsize(path) / 1048576}
scores = {}
for k in VARIANTS: scores[k] = judge(f'onnx/v-{k}.onnx')[1]; print('판', k, {a: round(b, 4) for a, b in scores[k].items()}, flush=True)
ok_ = [k for k in VARIANTS if scores[k]['mb'] < 240] or list(VARIANTS); pick = min(ok_, key=lambda k: scores[k]['kl']); print('고른 판', pick)
import shutil; shutil.copy(f'onnx/v-{pick}.onnx', q); s = ort.InferenceSession(q)
lo = s.run(['logits'], feeds)[0][0]
agree = float(np.mean(np.argmax(lo, -1) == np.argmax(lt, -1))); last_t = lt[-1]; last_o = lo[-1]
top = lambda v: [[int(i), round(float(v[i]), 3)] for i in np.argsort(-v)[:10]]
print('위치별 1위 일치율 (int8 vs fp32)', round(agree, 4), '· 마지막 상위 5', top(last_o)[:5], top(last_t)[:5])
ref = {'variants': scores, 'pick': pick, 'model': MID, 'vocab': VOCAB_SIZE, 'layers': nl, 'heads': nh, 'head_dim': hd, 'tokens': toks, 'top_fp32': top(last_t), 'top_int8': top(last_o), 'agree': agree, 'inputs': names[:4], 'fp32_onnx_vs_torch': {'full': ref_fp32[0], 'kv': ref_fp32[1]}}
# 공식 generate (fp32, top_p 0.95) — 4~8초를 멜로디만 비워 채우기 (베이스 · 앞 멜로디는 조건)
torch.manual_seed(0)
inputs = ops.clip(ev, 0, 8, clip_duration=False)
melody_after = [x for x in zip(ev[0::3], ev[1::3], ev[2::3]) if x[0] - TIME_OFFSET >= 400 and (x[2] - NOTE_OFFSET) // 128 == 0]
hist = [v for x in zip(ev[0::3], ev[1::3], ev[2::3]) if not (x[0] - TIME_OFFSET >= 400 and (x[2] - NOTE_OFFSET) // 128 == 0) for v in x]
bass = [v for x in zip(hist[0::3], hist[1::3], hist[2::3]) if (x[2] - NOTE_OFFSET) // 128 == 33 for v in x]
mel_before = [v for x in zip(hist[0::3], hist[1::3], hist[2::3]) if (x[2] - NOTE_OFFSET) // 128 == 0 for v in x]
out = generate(m, 4, 8, inputs=mel_before, controls=[t + CONTROL_OFFSET for t in bass], top_p=0.95)
gen = [(t - TIME_OFFSET, d - DUR_OFFSET, (n - NOTE_OFFSET) // 128, (n - NOTE_OFFSET) % 128) for t, d, n in zip(out[0::3], out[1::3], out[2::3]) if t - TIME_OFFSET >= 400]
print('공식 generate 4~8초 채운 사건', len(gen), gen[:10]); ref['gen_example'] = gen[:20]
# ---- 조각 나누기 ----
b = open(q, 'rb').read(); sha = hashlib.sha256(b).hexdigest(); CH = 45 * 1048576; parts = []
for k in range(0, len(b), CH):
    fn = f'amt-small-q.part{k // CH}'; open(f'{OUT}/{fn}', 'wb').write(b[k:k + CH]); parts.append({'file': fn, 'sha256': hashlib.sha256(b[k:k + CH]).hexdigest(), 'bytes': len(b[k:k + CH])})
os.remove(q)
meta = {'file': 'amt-small-q.onnx', 'mb': round(len(b) / 1048576, 2), 'sha256': sha, 'parts': parts}
print(json.dumps(meta, indent=1)); json.dump(meta, open(f'{OUT}/amt-meta.json', 'w'), indent=1); json.dump(ref, open(f'{OUT}/amt-ref.json', 'w'))
