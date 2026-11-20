# v1.1 ③ 박 · 음표 준비 (GitHub Actions 의 beat-prep 이 불러요)
#  · Beat This! (CPJKU, 코드 · 가중치 MIT): small0 · final0 체크포인트 → ONNX (입력 1×1500×128 로그 멜, 출력 beat · downbeat 로짓 1×1500), final0 은 int8 판도
#  · Basic Pitch (Spotify, Apache-2.0): 공식 nmp.onnx 를 그대로 거울에
#  · 기준값(ref): 수식으로 만든 시험 신호 — 앱의 JS 가 같은 수식으로 만들어 특징 · 로짓 · 박 · 음표를 비교해요
import json, os, sys, hashlib, numpy as np, torch
OUT = os.path.abspath('models'); os.makedirs(OUT, exist_ok=True)
SR = 22050

def lcg(seed=12345):
    s = seed
    while True:
        s = (s * 1664525 + 1013904223) % 4294967296
        yield s / 4294967296

def sig_beat(sec=16.0, bpm=120.0):
    """킥(매 박) · 스네어(2·4박) · 햇(8분) · 베이스(마디 첫 박) — 4/4, 정답 박 = k·0.5초"""
    n = int(SR * sec); t = np.arange(n) / SR; x = np.zeros(n); r = lcg(); noise = np.array([next(r) * 2 - 1 for _ in range(n)])
    beat = 60.0 / bpm
    for k in range(int(sec / beat)):
        t0 = k * beat; i0 = int(round(t0 * SR)); L = min(n - i0, int(0.35 * SR)); tt = np.arange(L) / SR
        x[i0:i0 + L] += 0.7 * np.exp(-tt * 18) * np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 30)) * tt)
        if k % 2 == 1:
            Ls = min(n - i0, int(0.2 * SR)); x[i0:i0 + Ls] += 0.35 * np.exp(-np.arange(Ls) / SR * 25) * noise[i0:i0 + Ls]
        if k % 4 == 0:
            Lb = min(n - i0, int(beat * 3.5 * SR)); tb = np.arange(Lb) / SR; x[i0:i0 + Lb] += 0.25 * np.sin(2 * np.pi * 55 * tb) * np.exp(-tb * 1.5)
        for h in (0, 0.5):
            ih = int(round((t0 + h * beat) * SR)); Lh = min(n - ih, int(0.04 * SR))
            if Lh > 0: x[ih:ih + Lh] += 0.12 * np.exp(-np.arange(Lh) / SR * 120) * noise[ih:ih + Lh]
    return x.astype(np.float64)

def sig_pitch():
    """도레미파솔라시도 0.5초씩 + 마지막 C 장3화음 1초 — 배음 4개(1/k), 짧은 어택 · 릴리스"""
    seq = [(i * 0.5, 0.45, p) for i, p in enumerate([60, 62, 64, 65, 67, 69, 71, 72])] + [(4.0, 1.0, p) for p in (60, 64, 67)]
    n = int(SR * 5.5); x = np.zeros(n)
    for t0, d, p in seq:
        f = 440.0 * 2 ** ((p - 69) / 12); i0 = int(round(t0 * SR)); L = int(d * SR); tt = np.arange(L) / SR
        env = np.minimum(1, tt / 0.01) * np.minimum(1, (d - tt) / 0.05)
        x[i0:i0 + L] += 0.2 * env * sum(np.sin(2 * np.pi * f * k * tt) / k for k in range(1, 5))
    return x.astype(np.float64), seq

ref = {'sr': SR, 'signals': {'beat': 'kick/snare/hat/bass 120bpm 16s, lcg 12345', 'pitch': 'C major scale 0.5s + C triad'}}
xb = sig_beat(); xp, seq = sig_pitch()
ref['truth'] = {'beats': [k * 0.5 for k in range(32)], 'downbeats': [k * 2.0 for k in range(8)], 'notes': [[a, a + d, p] for a, d, p in seq]}
ref['sig_check'] = {'beat_sum': float(np.abs(xb).sum()), 'beat_first': [float(v) for v in xb[:8]], 'beat_at1s': [float(v) for v in xb[SR:SR + 8]], 'pitch_sum': float(np.abs(xp).sum())}

# ---- Beat This! ----
from beat_this.inference import load_model, Audio2Beats
from beat_this.preprocessing import LogMelSpect
spec = LogMelSpect()(torch.tensor(xb, dtype=torch.float32)).numpy()
ref['mel'] = {'shape': list(spec.shape), 'first': [[float(v) for v in spec[f]] for f in range(3)], 'f100': [float(v) for v in spec[100]], 'mean': float(spec.mean())}

class Wrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        y = self.m(x); return y['beat'], y['downbeat']

import onnxruntime as ort
for ck in ['small0', 'final0']:
    m = load_model(ck, 'cpu'); w = Wrap(m).eval(); x = torch.randn(1, 1500, 128)
    path = f'{OUT}/beat-this-{ck}.onnx'
    with torch.no_grad():
        torch.onnx.export(w, x, path, opset_version=17, input_names=['spect'], output_names=['beat', 'downbeat'], dynamo=False)
        yb, yd = w(x)
    s = ort.InferenceSession(path); ob, od = s.run(None, {'spect': x.numpy()})
    print(ck, 'ONNX', os.path.getsize(path) // 1048576, 'MB · 최대 오차', float(np.abs(ob - yb.numpy()).max()), float(np.abs(od - yd.numpy()).max()))
    a2b = Audio2Beats(checkpoint_path=ck, device='cpu', dbn=False); beats, downs = a2b(xb, SR)
    sp = torch.tensor(spec)[None]; pad = torch.zeros(1, 1500, 128); pad[:, :sp.shape[1]] = sp
    with torch.no_grad(): lb, ld = w(pad)
    ref[ck] = {'beats': [float(v) for v in beats], 'downbeats': [float(v) for v in downs], 'logit_beat': [float(v) for v in lb[0, :sp.shape[1]]], 'logit_down': [float(v) for v in ld[0, :sp.shape[1]]]}
    print(ck, '박', len(beats), '다운비트', len(downs), [round(v, 2) for v in beats[:6]], [round(v, 2) for v in downs[:4]])
    if ck == 'final0':
        from onnxruntime.quantization import quantize_dynamic, QuantType
        qp = f'{OUT}/beat-this-final0-q.onnx'; quantize_dynamic(path, qp, weight_type=QuantType.QInt8)
        sq = ort.InferenceSession(qp); qb, qd = sq.run(None, {'spect': pad.numpy()})
        print('final0 int8', os.path.getsize(qp) // 1048576, 'MB · 로짓 오차(시험 신호)', float(np.abs(qb[0, :sp.shape[1]] - lb[0, :sp.shape[1]].numpy()).max()))

# ---- Basic Pitch (공식 nmp.onnx · 공식 후처리로 기준 음표) ----
import shutil, urllib.request
bp = f'{OUT}/basic-pitch-nmp.onnx'
urllib.request.urlretrieve('https://raw.githubusercontent.com/spotify/basic-pitch/main/basic_pitch/saved_models/icassp_2022/nmp.onnx', bp)
try:
    import soundfile as sf
    from basic_pitch.inference import predict, Model
    wav = os.path.abspath('pitch.wav'); sf.write(wav, xp.astype(np.float32), SR)
    out, midi, notes = predict(wav, Model(bp))
    ref['basic_pitch'] = {'notes': [[float(a), float(b), int(p), float(v)] for a, b, p, v, _ in notes], 'note_first': [[float(v) for v in out['note'][f]] for f in (10, 50)], 'frames': int(out['note'].shape[0])}
    print('Basic Pitch 음표', len(notes), sorted([(round(a, 2), int(p)) for a, b, p, v, _ in notes])[:12])
except Exception as e:
    print('Basic Pitch 기준값 실패 (모델 거울은 올려요):', e)

with open(f'{OUT}/beat-ref.json', 'w') as f: json.dump(ref, f)
meta = {}
for fn in sorted(os.listdir(OUT)):
    if fn.endswith('.onnx'): meta[fn] = {'mb': round(os.path.getsize(f'{OUT}/{fn}') / 1048576, 2), 'sha256': hashlib.sha256(open(f'{OUT}/{fn}', 'rb').read()).hexdigest()}
print(json.dumps(meta, indent=1))
with open(f'{OUT}/beat-meta.json', 'w') as f: json.dump(meta, f, indent=1)
