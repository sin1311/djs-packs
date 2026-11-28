# stdio 24장 V8-2e 받아쓰기 실측 — 영상처럼 어려운 말 (Actions · whisper-noise 워크플로). 공용 코드 whisper_common.py
#  ① 배경음악 위 말: FLEURS 시험 문장(CC BY 4.0)에 stdio 엔진이 만든 곡(evals/music — 우리 작품 · 저작권 걱정 없음)을 말 대 음악 +10 · +5 · 0 dB 로 섞음
#  ② 겹친 말: 같은 언어 다른 문장(다른 사람)을 0.3초 늦게 −6 · 0 dB 로 겹침 — 주인공 문장을 받아쓰는지
#  ③ 음원 분리 먼저: 0 dB 음악 위 말을 HTDemucs(demucs 패키지 · MIT · 앱의 htdemucs 와 같은 가중치)로 목소리만 뽑고 받아쓰기
#  tiny · base 둘 다 · 한국어 CER · 영어 WER · 빈 비율. 오디오는 이 작업 안에서만 · 숫자만 남김.
import os, sys, json, time, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from whisper_common import *
import numpy as np, soundfile as sf, scipy.signal as ss
R = np.random.default_rng(7)
beds = []
for f in sorted(glob.glob('evals/music/*.flac')):
    x, r = sf.read(f); x = x.mean(1) if x.ndim > 1 else x
    if r != 16000: x = ss.resample_poly(x, 16000, r)
    beds.append((os.path.basename(f)[:-5], x.astype(np.float32)))
print('음악', [(n, round(len(x) / 16000, 1)) for n, x in beds])
rms = lambda x: float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)
def with_music(x, snr, k):
    n, m = beds[k % len(beds)]; o = int(R.integers(0, max(1, len(m) - len(x))))
    seg = np.resize(m[o:o + len(x)], len(x)); g = rms(x) / rms(seg) / (10 ** (snr / 20))
    y = x + g * seg; return (y / max(1.0, np.abs(y).max() / 0.99)).astype(np.float32)
def with_talk(x, y, db):
    d = int(0.3 * 16000); z = np.zeros(max(len(x), len(y) + d), np.float32); z[:len(x)] += x
    z[d:d + len(y)] += y * (rms(x) / rms(y)) * (10 ** (db / 20)); return (z / max(1.0, np.abs(z).max() / 0.99)).astype(np.float32)
SEP = None
def vocals(x):
    global SEP
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    if SEP is None: SEP = get_model('htdemucs'); SEP.eval()
    y = ss.resample_poly(x, 44100, 16000).astype(np.float32); t = torch.from_numpy(np.stack([y, y]))[None]
    with torch.no_grad(): out = apply_model(SEP, t, shifts=0, overlap=0.25, progress=False)[0]
    v = out[SEP.sources.index('vocals')].mean(0).numpy(); return ss.resample_poly(v, 16000, 44100).astype(np.float32)
data = {'ko': fleurs('ko_kr'), 'en': fleurs('en_us')}
CONDS = ['clean', 'music+10', 'music+5', 'music0', 'sep0', 'talk-6', 'talk0']
inputs = {}
for lk, items in data.items():
    for c in CONDS:
        xs = []
        for i, (x, ref) in enumerate(items):
            if c == 'clean': y = x
            elif c.startswith('music'): y = with_music(x, float(c[5:]), i)
            elif c == 'sep0': y = vocals(with_music(x, 0.0, i))
            else: y = with_talk(x, items[(i + 1) % len(items)][0], float(c[4:]))
            xs.append((y, ref))
        inputs[(lk, c)] = xs
    print('입력 준비', lk, flush=True)
res = {'fleurs': 'google/fleurs (CC BY 4.0) test split', 'music': 'stdio 엔진이 만든 곡(evals/music)', 'n': N, 'by': {}}
for mk, (enc, dec, h, dm) in MODELS.items():
    M = Model(enc, dec, h, dm)
    for (lk, c), items in inputs.items():
        errs = tot = empty = 0
        for x, ref in items:
            r = M.run(x); hyp, rf = norm(r['text']), norm(ref)
            a, b = (hyp.replace(' ', ''), rf.replace(' ', '')) if lk == 'ko' else (hyp.split(), rf.split())
            errs += lev(a, b); tot += max(1, len(b)); empty += not r['text']
        res['by'][f'{mk}-{lk}-{c}'] = {'err': round(errs / tot, 4), 'empty': round(empty / len(items), 3)}
        print(mk, lk, c, res['by'][f'{mk}-{lk}-{c}'], flush=True)
os.makedirs('models', exist_ok=True); json.dump(res, open('models/whisper-noise.json', 'w'), ensure_ascii=False, indent=1)
tab = {f'{m}-{l}': {c: res['by'][f'{m}-{l}-{c}']['err'] for c in CONDS} for m in MODELS for l in data}
print('RESULT', json.dumps(tab, ensure_ascii=False))
