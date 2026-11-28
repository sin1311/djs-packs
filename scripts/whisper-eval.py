# stdio 24장 V8-2d 받아쓰기 실측 (Actions · whisper-eval 워크플로): 진짜 사람 말(Google FLEURS 시험 문장 · CC BY 4.0)로
#  앱과 같은 int8 ONNX · 같은 탐욕 풀기 규칙(whisper-prep.py rules 와 같음) → 한국어 글자 오류율(CER) · 영어 단어 오류율(WER) · 언어 맞힘 · 문장 끝 부호 · 지어냄 · 시간.
#  tiny(앱이 지금 쓰는 것 — djs-packs models 가지 고정 커밋) vs base(onnx-community/whisper-base int8 — 바꿀 값어치가 있는지).
#  오디오는 이 작업 안에서만 쓰고 숫자만 남겨요(FLEURS 는 CC BY 4.0 — 출처 표시는 결과 JSON · 앱 문서에).
import json, os, sys, io, tarfile, time, re, unicodedata, urllib.request
import numpy as np
N = int(os.environ.get('N', '40'))
from huggingface_hub import hf_hub_download, HfApi
import onnxruntime as ort, soundfile as sf, scipy.signal as ss
from transformers import WhisperFeatureExtractor
FE = WhisperFeatureExtractor.from_pretrained('openai/whisper-tiny')

def get(u):
    with urllib.request.urlopen(urllib.request.Request(u, headers={'user-agent': 'djs-packs-whisper-eval'}), timeout=300) as r: return r.read()
MC = 'd750af053d58f9ecf153fa13c5ef2ca9e9cd069e'  # 앱이 고정한 models 커밋
os.makedirs('we', exist_ok=True)
for f in ['whisper-tiny-enc.onnx', 'whisper-tiny-dec.onnx', 'whisper-gen.json', 'whisper-vocab.json']:
    open('we/' + f, 'wb').write(get(f'https://raw.githubusercontent.com/sin1311/djs-packs/{MC}/{f}'))
G = json.load(open('we/whisper-gen.json')); toks = json.load(open('we/whisper-vocab.json'))
SOT, EOT, TR, NOTS, TS0 = G['sot'], G['eot'], G['transcribe'], G['notimestamps'], G['ts0']
LANG = G['lang']
binfo = HfApi().model_info('onnx-community/whisper-base'); BREV = binfo.sha
MODELS = { 'tiny': ('we/whisper-tiny-enc.onnx', 'we/whisper-tiny-dec.onnx', 6, 384),
  'base': (hf_hub_download('onnx-community/whisper-base', 'onnx/encoder_model_quantized.onnx', revision=BREV), hf_hub_download('onnx-community/whisper-base', 'onnx/decoder_model_merged_quantized.onnx', revision=BREV), 8, 512) }
print('base 고정', BREV, {k: [os.path.getsize(p) for p in v[:2]] for k, v in MODELS.items()})

def lsm(x): m = x.max(); return x - m - np.log(np.exp(x - m).sum())
def rules(logits, seq, k):
    x = logits.astype(np.float64).copy(); NEG = -1e9
    for t in G['suppress']: x[t] = NEG
    x[NOTS] = NEG; x[EOT + 1:TS0] = NEG
    if k == 0:
        for t in G['beginSuppress']: x[t] = NEG
    last_ts = len(seq) >= 1 and seq[-1] >= TS0; pen_ts = len(seq) < 2 or seq[-2] >= TS0
    if last_ts:
        if pen_ts: x[TS0:] = NEG
        else: x[:EOT] = NEG
    tss = [t for t in seq if t >= TS0]
    if tss: x[TS0:(tss[-1] if (last_ts and not pen_ts) else tss[-1] + 1)] = NEG
    if k == 0: x[:TS0] = NEG; x[TS0 + G['maxInitialTs'] + 1:] = NEG
    lp = lsm(x); ts_lp = np.log(np.exp(lp[TS0:] - lp[TS0:].max()).sum()) + lp[TS0:].max()
    if ts_lp > lp[:TS0].max(): x[:TS0] = NEG
    return int(np.argmax(x))
def inv():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}
INV = inv()
def text_of(seq): return bytes(INV.get(ch, 32) for t in seq if t < EOT for ch in toks[t]).decode('utf8', 'replace').strip()
class Model:
    def __init__(s, enc, dec, heads, dm):
        so = ort.SessionOptions(); so.intra_op_num_threads = 2
        s.E = ort.InferenceSession(enc, so, providers=['CPUExecutionProvider']); s.D = ort.InferenceSession(dec, so, providers=['CPUExecutionProvider'])
        s.din = [i.name for i in s.D.get_inputs()]; s.dout = [o.name for o in s.D.get_outputs()]; s.h = heads; s.hd = dm // heads
    def step(s, ids, eh, past, cache):
        f = {'input_ids': np.array([ids], np.int64), 'encoder_hidden_states': eh}
        for n in s.din:
            if n.startswith('past_key_values'): f[n] = past.get(n, np.zeros((1, s.h, 0, s.hd), np.float32))
        if 'use_cache_branch' in s.din: f['use_cache_branch'] = np.array([cache])
        o = s.D.run(None, f); nxt = {n.replace('present', 'past_key_values'): v for n, v in zip(s.dout, o) if n.startswith('present')}
        return o[0][0, -1], o[0][0, 0], nxt
    def run(s, x, maxn=224):
        feat = FE(x, sampling_rate=16000, return_tensors='np').input_features.astype(np.float32); eh = s.E.run(None, {s.E.get_inputs()[0].name: feat})[0]
        lg, _, _ = s.step([SOT], eh, {}, False); ids = list(LANG.values()); lid = ids[int(np.argmax(lg[ids]))]
        logits, first, past = s.step([SOT, lid, TR], eh, {}, False); enc = {k: v for k, v in past.items() if '.encoder.' in k}
        e = np.exp(first - first.max()); nos = float(e[NOTS - 1] / e.sum())
        seq = []; lps = 0.0; loop = False
        for k in range(maxn):
            t = rules(logits, seq, k)
            if t == EOT: break
            seq.append(t); lps += float(lsm(logits.astype(np.float64))[t])
            if len(seq) >= 8 and all(v == t for v in seq[-8:]): loop = True; break
            logits, _, past = s.step([t], eh, {**{a: b for a, b in past.items() if '.encoder.' not in a}, **enc}, True)
            past = {**{a: b for a, b in past.items() if '.encoder.' not in a}, **enc}
        avg = lps / max(1, len(seq)); silent = nos > 0.6 and avg < -1
        lang = [k for k, v in LANG.items() if v == lid][0].strip('<|>')
        return {'text': '' if silent or loop else text_of(seq), 'lang': lang, 'loop': loop, 'silent': silent, 'n': len(seq)}

def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1): p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[len(b)]
def norm(t): t = unicodedata.normalize('NFKC', t).lower(); return re.sub(r'[^\w\s]', '', t)
def fleurs(lang):
    tsv = open(hf_hub_download('google/fleurs', f'data/{lang}/test.tsv', repo_type='dataset'), encoding='utf8').read().strip().split('\n')
    rows = [r.split('\t') for r in tsv][:N]; want = {r[1]: r[2] for r in rows}
    tf = tarfile.open(hf_hub_download('google/fleurs', f'data/{lang}/audio/test.tar.gz', repo_type='dataset'))
    out = []
    for m in tf.getmembers():
        b = os.path.basename(m.name)
        if b in want and m.isfile():
            x, r = sf.read(io.BytesIO(tf.extractfile(m).read())); x = x.mean(1) if x.ndim > 1 else x
            if r != 16000: x = ss.resample_poly(x, 16000, r)
            out.append((x.astype(np.float32), want[b]))
        if len(out) >= N: break
    return out
res = {'fleurs': 'google/fleurs (CC BY 4.0) test split', 'n': N, 'models': {'tiny': MC, 'base': BREV}, 'by': {}}
data = {'ko': fleurs('ko_kr'), 'en': fleurs('en_us')}
for mk, (enc, dec, h, dm) in MODELS.items():
    M = Model(enc, dec, h, dm)
    for lk, items in data.items():
        errs = tot = 0; lang_ok = ends = halluc = 0; secs = 0.0; dur = 0.0; ex = []
        for x, ref in items:
            t0 = time.time(); r = M.run(x); secs += time.time() - t0; dur += len(x) / 16000
            hyp, rf = norm(r['text']), norm(ref)
            if lk == 'ko': a, b = hyp.replace(' ', ''), rf.replace(' ', '')
            else: a, b = hyp.split(), rf.split()
            errs += lev(a, b); tot += max(1, len(b)); lang_ok += r['lang'] == lk; ends += bool(re.search(r'[.?!。…]\s*$', r['text'])); halluc += (r['loop'] or r['silent'] or not r['text'])
            if len(ex) < 4: ex.append({'ref': ref[:80], 'hyp': r['text'][:80], 'lang': r['lang']})
        res['by'][f'{mk}-{lk}'] = {'err': round(errs / tot, 4), 'metric': 'CER' if lk == 'ko' else 'WER', 'langOk': round(lang_ok / len(items), 3), 'endPunct': round(ends / len(items), 3), 'empty': round(halluc / len(items), 3), 'rtf': round(secs / dur, 3), 'n': len(items), 'ex': ex}
        print(mk, lk, json.dumps(res['by'][f'{mk}-{lk}'], ensure_ascii=False))
os.makedirs('models', exist_ok=True); json.dump(res, open('models/whisper-eval.json', 'w'), ensure_ascii=False, indent=1)
print('RESULT', json.dumps({k: {a: v[a] for a in ['err', 'metric', 'langOk', 'endPunct', 'empty', 'rtf']} for k, v in res['by'].items()}, ensure_ascii=False))
