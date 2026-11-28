# stdio 24장 V8-2 말 받아쓰기 준비 (Actions · whisper-prep 워크플로): Whisper tiny(다국어) ONNX(int8) + 어휘 + 기준값.
#  원본: openai/whisper(코드 MIT) · 가중치 openai/whisper-tiny(모델 카드 license: apache-2.0) · ONNX 판 onnx-community/whisper-tiny(같은 카드) — 원문 licenses/audio/ (license-check 묶음 audio)
#  앱은 소리 → 로그 멜(80 띠 · 30초 3000 프레임)을 직접 만들고(src/ai/whisper.js) 인코더 · 디코더(KV 캐시 합본)는 ONNX 로 돌려요.
#  기준값(whisper-ref.json): ① transformers WhisperFeatureExtractor 로그 멜(앱 전처리 확인) ② 같은 ONNX 로 탐욕 풀기(시간 표시 규칙 — 앱과 같은 식) 토큰 · 글
#  ③ PyTorch 원본 파이프라인 글(정답에 가까운 것 — int8 이 얼마나 비슷한지).
import json, hashlib, os, subprocess, base64
import numpy as np

OUT = 'models'; os.makedirs(OUT, exist_ok=True); os.makedirs('wh', exist_ok=True)
sha = lambda b: hashlib.sha256(b).hexdigest()
from huggingface_hub import HfApi, hf_hub_download
SIZE = os.environ.get('SIZE', 'tiny')  # tiny(T1) · base(T2 — 실측 whisper-eval: 한국어 CER 24% → 10%)
REPO = f'onnx-community/whisper-{SIZE}'
PFX = 'whisper' if SIZE == 'tiny' else f'whisper-{SIZE}'  # tiny 는 예전 이름 그대로(앱이 이미 고정)
info = HfApi().model_info(REPO); REV = info.sha; print('onnx-community/whisper-tiny 고정', REV)
files = sorted(s.rfilename for s in info.siblings); print('\n'.join(f for f in files if f.startswith('onnx/') or f.endswith('.json')))
def dl(f): return hf_hub_download(REPO, f, revision=REV)
ENC, DEC = dl('onnx/encoder_model_quantized.onnx'), dl('onnx/decoder_model_merged_quantized.onnx')
TOK, GEN, CFG, PRE = (json.load(open(dl(f))) for f in ['tokenizer.json', 'generation_config.json', 'config.json', 'preprocessor_config.json'])
card = open(dl('README.md'), encoding='utf8').read()[:1500]

# 어휘: id → 바이트 수준 BPE 글자(GPT-2 bytes_to_unicode 표현 그대로 — 앱이 바이트로 되돌린 뒤 UTF-8 로 읽음) · 더한 토큰(특수)
vocab = TOK['model']['vocab']; V = max(max(vocab.values()), max(t['id'] for t in TOK['added_tokens'])) + 1
toks = [''] * V
for s, i in vocab.items(): toks[i] = s
special = {}
for t in TOK['added_tokens']: toks[t['id']] = t['content']; special[t['content']] = t['id']
SOT, EOT = special['<|startoftranscript|>'], special['<|endoftext|>']
TR, NOTS = special['<|transcribe|>'], special['<|notimestamps|>']
TS0 = special.get('<|0.00|>') or (NOTS + 1)
LANG = {k: v for k, v in GEN['lang_to_id'].items()}
gen = { 'rev': REV, 'sot': SOT, 'eot': EOT, 'transcribe': TR, 'translate': special['<|translate|>'], 'notimestamps': NOTS, 'ts0': TS0, 'lang': LANG,
  'suppress': GEN.get('suppress_tokens', []), 'beginSuppress': GEN.get('begin_suppress_tokens', [220, EOT]), 'maxInitialTs': GEN.get('max_initial_timestamp_index', 50),
  'nLayers': CFG['decoder_layers'], 'nHeads': CFG['decoder_attention_heads'], 'dModel': CFG['d_model'], 'maxTarget': CFG.get('max_target_positions', 448),
  'pre': {k: PRE[k] for k in ['feature_size', 'sampling_rate', 'hop_length', 'n_fft', 'chunk_length', 'n_samples', 'nb_max_frames'] if k in PRE} }

import onnxruntime as ort
so = ort.SessionOptions(); so.intra_op_num_threads = 2
E = ort.InferenceSession(ENC, so, providers=['CPUExecutionProvider']); D = ort.InferenceSession(DEC, so, providers=['CPUExecutionProvider'])
io = { 'enc': { 'in': [i.name for i in E.get_inputs()], 'out': [o.name for o in E.get_outputs()] }, 'dec': { 'in': [[i.name, i.type, i.shape] for i in D.get_inputs()], 'out': [o.name for o in D.get_outputs()] } }
print(json.dumps(io)[:1500]); gen['io'] = io

from transformers import WhisperFeatureExtractor
FE = WhisperFeatureExtractor.from_pretrained(f'openai/whisper-{SIZE}')

def lsm(x): m = x.max(); return x - m - np.log(np.exp(x - m).sum())
def rules(logits, seq, k):
    """앱(src/ai/whisper.js)과 같은 규칙 — openai/whisper ApplyTimestampRules 를 따라 쓴 것(탐욕 · 한 줄)"""
    x = logits.astype(np.float64).copy(); NEG = -1e9
    for t in gen['suppress']: x[t] = NEG
    x[NOTS] = NEG
    x[EOT + 1:TS0] = NEG                    # 특수 토큰(언어 · 할 일 · 이전 문맥 …)은 글 가운데 안 나오게
    if k == 0:
        for t in gen['beginSuppress']: x[t] = NEG
    out = seq
    last_ts = len(out) >= 1 and out[-1] >= TS0
    pen_ts = len(out) < 2 or out[-2] >= TS0
    if last_ts:
        if pen_ts: x[TS0:] = NEG            # 짝이 끝났으면 다음은 글
        else: x[:EOT] = NEG                  # 짝 가운데면 다음은 시간 표시 또는 끝
    tss = [t for t in out if t >= TS0]
    if tss:
        lo = tss[-1] if (last_ts and not pen_ts) else tss[-1] + 1   # 시간은 되돌아가지 않음
        x[TS0:lo] = NEG
    if k == 0:
        x[:TS0] = NEG                        # 첫 토큰은 시간 표시
        x[TS0 + gen['maxInitialTs'] + 1:] = NEG
    lp = lsm(x)
    ts_lp = np.log(np.exp(lp[TS0:] - lp[TS0:].max()).sum()) + lp[TS0:].max()
    if ts_lp > lp[:TS0].max(): x[:TS0] = NEG   # 시간 표시 쪽 확률 합이 가장 큰 글보다 크면 시간 표시
    return int(np.argmax(x))

def past0():
    feeds = {}
    for name, typ, shape in io['dec']['in']:
        if name.startswith('past_key_values'): feeds[name] = np.zeros((1, gen['nHeads'], 0, gen['dModel'] // gen['nHeads']), np.float32)
    return feeds

def decode(feat, lang=None, maxn=224):
    eh = E.run(None, {io['enc']['in'][0]: feat})[0]
    def step(ids, past, cache):
        f = {'input_ids': np.array([ids], np.int64), 'encoder_hidden_states': eh, **past}
        if 'use_cache_branch' in [n for n, _, _ in io['dec']['in']]: f['use_cache_branch'] = np.array([cache])
        o = D.run(None, f); names = io['dec']['out']
        newp = {}
        for n, v in zip(names, o):
            if n.startswith('present'):
                pk = n.replace('present', 'past_key_values')
                newp[pk] = v
        return o[0][0, -1], newp
    # 언어 찾기: [SOT] 한 번 → 언어 토큰 중 가장 큰 것
    if lang is None:
        lg, _ = step([SOT], past0(), False); ids = list(LANG.values()); lang_id = ids[int(np.argmax(lg[ids]))]
    else: lang_id = LANG[lang]
    prompt = [SOT, lang_id, TR]
    logits, past = step(prompt, past0(), False)
    enc_past = {k: v for k, v in past.items() if '.encoder.' in k}
    seq = []
    for k in range(maxn):
        t = rules(logits, seq, k)
        if t == EOT: break
        seq.append(t)
        p2 = {**past, **enc_past}
        logits, past = step([t], p2, True)
        past = {**{k2: v for k2, v in past.items() if '.encoder.' not in k2}, **enc_past}  # 캐시 가지에선 인코더 쪽 키 · 값이 비어 나와요 → 첫 걸음 것을 계속 씀
    return prompt, seq

def textOf(seq):
    bs = bytes_to_unicode_inv(); out = []; segs = []; cur = []; t0 = None
    for t in seq:
        if t >= TS0:
            v = round((t - TS0) * 0.02, 2)
            if t0 is None: t0 = v
            else: segs.append({'t0': t0, 't1': v, 'text': bdec(cur, bs)}); cur = []; t0 = None
        elif t < EOT: cur.append(t)
    if cur: segs.append({'t0': t0, 't1': None, 'text': bdec(cur, bs)})
    return segs
def bytes_to_unicode_inv():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1)); cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}
def bdec(ids, inv): return bytes(b for i in ids for ch in toks[i] for b in [inv.get(ch, 32)]).decode('utf8', 'replace')

# 시험 소리(16 kHz): ① 말 1 ② 말 2 ③ 말 1 + 1.5초 조용 + 말 2 ④ 조용 + 낮은 잡음(말 없음) ⑤ 440 Hz 사인 2초
SR = 16000
def flite(t, f):
    subprocess.run(['flite', '-voice', 'slt', '-t', t, '-o', 'wh/' + f], check=True)
    import soundfile as sf, scipy.signal as ss
    x, r = sf.read('wh/' + f); x = x.astype(np.float64)
    if x.ndim > 1: x = x.mean(1)
    if r != SR: x = ss.resample_poly(x, SR, r)
    return x.astype(np.float32)
s1, s2 = flite('Tonight, we drive into the city.', 'a.wav'), flite('This is where the night begins.', 'b.wav')
def lcg(seed):
    a = [seed]
    def r(): a[0] = (a[0] * 1664525 + 1013904223) % 4294967296; return a[0] / 4294967296
    return r
r = lcg(7)
q16 = lambda x: (np.round(np.clip(x, -1, 1) * 32767).astype(np.int16).astype(np.float32) / 32767)  # 16비트로 저장해 앱에 주니 기준값도 같은 16비트 소리로(앱 = 파이썬 입력이 표본까지 같게)
s1, s2 = q16(s1), q16(s2)
sig = { 'say1': s1, 'say2': s2, 'two': np.concatenate([s1, np.zeros(int(1.5 * SR), np.float32), s2]),
  'quiet': np.array([0.003 * (r() * 2 - 1) for _ in range(SR * 3)], np.float32), 'sine': (0.3 * np.sin(2 * np.pi * 440 * np.arange(SR * 2) / SR)).astype(np.float32) }

import torch
from transformers import pipeline
asr = pipeline('automatic-speech-recognition', model=f'openai/whisper-{SIZE}', device='cpu')
ref = {'repo': REPO, 'rev': REV, 'sr': SR, 'cases': {}}
for name, x in sig.items():
    feat = FE(x, sampling_rate=SR, return_tensors='np').input_features.astype(np.float32)  # [1, 80, 3000]
    F = feat[0]
    prompt, seq = decode(feat)
    pt = asr({'raw': x.astype(np.float32), 'sampling_rate': SR}, return_timestamps=True, generate_kwargs={'task': 'transcribe'})
    ref['cases'][name] = { 'n': int(len(x)), 'wav': base64.b64encode(np.round(np.clip(x, -1, 1) * 32767).astype('<i2').tobytes()).decode() if name in ('say1', 'say2', 'two') else None,
      'melRows': {str(i): [round(float(v), 5) for v in F[:, i]] for i in [0, 10, 50, 120, 299]}, 'melSum': round(float(F.sum()), 3), 'melMax': round(float(F.max()), 5), 'melMin': round(float(F.min()), 5),
      'prompt': prompt, 'tokens': seq, 'segs': textOf(seq), 'torch': {'text': pt['text'], 'chunks': [{'t': list(c['timestamp']), 'text': c['text']} for c in pt.get('chunks', [])]} }
    print(name, prompt, len(seq), textOf(seq), '| torch:', pt['text'])

def put(src, dst):
    b = open(src, 'rb').read(); open(f'{OUT}/{dst}', 'wb').write(b); return {'file': dst, 'bytes': len(b), 'sha256': sha(b)}
man = { 'model': REPO, 'rev': REV, 'license': 'apache-2.0 (모델 카드) · MIT (openai/whisper 코드)', 'card_head': card,
  'enc': put(ENC, f'whisper-{SIZE}-enc.onnx'), 'dec': put(DEC, f'whisper-{SIZE}-dec.onnx') }
vb = json.dumps(toks, ensure_ascii=False).encode(); print('어휘 sha', sha(vb), '(tiny 와 같아야 해요 — 다국어 어휘 하나)')
if SIZE == 'tiny': open(f'{OUT}/whisper-vocab.json', 'wb').write(vb)
json.dump({**gen, 'files': man, 'size': SIZE, 'vocabSha': sha(vb)}, open(f'{OUT}/{PFX}-gen.json', 'w'), ensure_ascii=False, indent=1)
json.dump(ref, open(f'{OUT}/{PFX}-ref.json', 'w'), ensure_ascii=False)
for f in ([f'{PFX}-gen.json', f'{PFX}-ref.json'] + (['whisper-vocab.json'] if SIZE == 'tiny' else [])):
    b = open(f'{OUT}/{f}', 'rb').read(); print(f, len(b), sha(b))
print(json.dumps(man, ensure_ascii=False)[:600])
