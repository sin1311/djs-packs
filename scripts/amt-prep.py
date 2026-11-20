# v1.2 ① AMT 준비 (GitHub Actions 의 amt-prep 이 불러요)
#  · Anticipatory Music Transformer small (stanford-crfm/music-small-800k · 코드 Apache-2.0 · 가중치 라이선스는 모델 카드에서 확인해 찍어요)
#  · optimum 으로 ONNX(KV 캐시) → int8 동적 양자화 → 45MB 조각으로 (GitHub 파일 100MB · jsDelivr 50MB 한도)
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
subprocess.run(['optimum-cli', 'export', 'onnx', '--model', MID, '--task', 'text-generation-with-past', '--opset', '17', 'onnx'], check=True)
src = [f for f in os.listdir('onnx') if f.endswith('.onnx')]; print('내보낸 파일', src)
fp = 'onnx/model.onnx' if 'model.onnx' in src else 'onnx/decoder_model_merged.onnx'
from onnxruntime.quantization import quantize_dynamic, QuantType
q = f'{OUT}/amt-small-q.onnx'
quantize_dynamic(fp, q, weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul', 'Gemm', 'Gather'])
print('int8', os.path.getsize(q) // 1048576, 'MB')
import onnxruntime as ort
s = ort.InferenceSession(q); print('입력', [i.name for i in s.get_inputs()][:6], '…', len(s.get_inputs()), '· 출력', [o.name for o in s.get_outputs()][:3])
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
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(MID).eval()
with torch.no_grad(): lt = m(torch.tensor([toks])).logits[0].numpy()
names = [i.name for i in s.get_inputs()]; L = len(toks); feeds = {'input_ids': np.array([toks], dtype=np.int64)}
if 'attention_mask' in names: feeds['attention_mask'] = np.ones((1, L), dtype=np.int64)
if 'position_ids' in names: feeds['position_ids'] = np.arange(L, dtype=np.int64)[None]
cfg = m.config; nl, nh, hd = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head
for n in names:
    if n.startswith('past_key_values'): feeds[n] = np.zeros((1, nh, 0, hd), dtype=np.float32)
if 'use_cache_branch' in names: feeds['use_cache_branch'] = np.array([False])
lo = s.run(['logits'], feeds)[0][0]
agree = float(np.mean(np.argmax(lo, -1) == np.argmax(lt, -1))); last_t = lt[-1]; last_o = lo[-1]
top = lambda v: [[int(i), round(float(v[i]), 3)] for i in np.argsort(-v)[:10]]
print('위치별 1위 일치율 (int8 vs fp32)', round(agree, 4), '· 마지막 상위 5', top(last_o)[:5], top(last_t)[:5])
ref = {'model': MID, 'vocab': VOCAB_SIZE, 'layers': nl, 'heads': nh, 'head_dim': hd, 'tokens': toks, 'top_fp32': top(last_t), 'top_int8': top(last_o), 'agree': agree, 'inputs': names[:4]}
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
