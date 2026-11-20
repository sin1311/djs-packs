# v1.1 ④-e 음원 분리 준비 (GitHub Actions 의 sep-prep 이 불러요)
#  · HTDemucs (Meta, 코드 · 가중치 MIT) — demucs.onnx(sevagh, MIT)의 방식대로 STFT/ISTFT 를 모델 밖으로 뺀 판으로 ONNX 변환
#    입력 mix [1,2,343980] (44.1kHz 7.8초) · spec [1,4,2048,336] (복소를 채널로) → 출력 x [1,4,4,2048,336] (주파수 가지) · xt [1,4,2,343980] (시간 가지)
#  · 기준값: 정답 스템을 아는 합성 혼합음(드럼 · 베이스 · 코드 · 멜로디, CC0)을 공식 경로(apply_model, shifts=0, overlap 0.25)로 분리 → SDR · 대조 표본
import os, sys, json, hashlib, math, numpy as np, torch
sys.path.insert(0, os.path.abspath('demucs.onnx/demucs-for-onnx'))
from demucs.pretrained import get_model
from demucs.htdemucs import HTDemucs, standalone_spec, standalone_magnitude
from demucs.apply import apply_model
OUT = os.path.abspath('models'); os.makedirs(OUT, exist_ok=True)
SR = 44100

bag = get_model('htdemucs'); core = bag.models[0] if hasattr(bag, 'models') else bag
assert isinstance(core, HTDemucs); core.eval()
print('소스', core.sources, '· segment', float(core.segment), '· 표본율', core.samplerate)
TL = int(core.segment * core.samplerate)
mix = torch.randn(1, 2, TL); spec = standalone_magnitude(standalone_spec(mix))
path = f'{OUT}/htdemucs.onnx'
with torch.no_grad():
    torch.onnx.export(core, (mix, spec), path, export_params=True, opset_version=17, do_constant_folding=True, input_names=['mix', 'spec'], output_names=['x', 'xt'], dynamo=False)
    yx, yxt = core(mix, spec)
import onnxruntime as ort
s = ort.InferenceSession(path); ox, oxt = s.run(None, {'mix': mix.numpy(), 'spec': spec.numpy()})
print('ONNX', os.path.getsize(path) // 1048576, 'MB · 최대 오차', float(np.abs(ox - yx.numpy()).max()), float(np.abs(oxt - yxt.numpy()).max()))
try:
    from onnxconverter_common import float16
    import onnx
    m16 = float16.convert_float_to_float16(onnx.load(path), keep_io_types=True)
    onnx.save(m16, f'{OUT}/htdemucs-f16.onnx'); s16 = ort.InferenceSession(f'{OUT}/htdemucs-f16.onnx'); hx, hxt = s16.run(None, {'mix': mix.numpy(), 'spec': spec.numpy()})
    print('f16', os.path.getsize(f'{OUT}/htdemucs-f16.onnx') // 1048576, 'MB · 오차', float(np.abs(hx - ox).max()), float(np.abs(hxt - oxt).max()))
except Exception as e:
    print('f16 실패:', e)

# ---- 합성 혼합음 (정답 스템) ----
def lcg(seed):
    st = seed
    while True:
        st = (st * 1664525 + 1013904223) % 4294967296; yield st / 4294967296
sec, bpm = 12.0, 112.0; n = int(SR * sec); t = np.arange(n) / SR; beat = 60 / bpm; r = lcg(7)
stems = {k: np.zeros((2, n)) for k in ['drums', 'bass', 'other', 'vocals']}
noise = np.array([next(r) * 2 - 1 for _ in range(n)])
def tone(ch, t0, d, f, a, k=6, pan=0.0):
    i0 = int(t0 * SR); L = min(n - i0, int(d * SR)); tt = np.arange(L) / SR; env = np.minimum(1, tt / 0.01) * np.minimum(1, (d - tt) / 0.05)
    v = a * env * sum(np.sin(2 * np.pi * f * h * tt) / h for h in range(1, k + 1)); ch[0, i0:i0 + L] += v * (1 - pan) ; ch[1, i0:i0 + L] += v * (1 + pan)
hz = lambda m: 440 * 2 ** ((m - 69) / 12)
CH = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]
for k in range(int(sec / beat)):
    t0 = k * beat; i0 = int(t0 * SR); L = min(n - i0, int(0.3 * SR)); tt = np.arange(L) / SR
    kick = 0.7 * np.exp(-tt * 18) * np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 30)) * tt); stems['drums'][:, i0:i0 + L] += kick
    if k % 2 == 1:
        Ls = min(n - i0, int(0.2 * SR)); sn = 0.3 * np.exp(-np.arange(Ls) / SR * 25) * noise[i0:i0 + Ls]; stems['drums'][:, i0:i0 + Ls] += sn
    for h in (0, 0.5):
        ih = int((t0 + h * beat) * SR); Lh = min(n - ih, int(0.04 * SR)); stems['drums'][:, ih:ih + Lh] += 0.08 * np.exp(-np.arange(Lh) / SR * 120) * noise[ih:ih + Lh]
    bar = k // 4; ch = CH[bar % 4]
    if k % 4 == 0:
        tone(stems['bass'], t0, beat * 3.8, hz(ch[0] - 24), 0.25, 3)
        for m in ch: tone(stems['other'], t0, beat * 3.9, hz(m), 0.05, 6, pan=0.3)
    mel = [72, 74, 76, 79, 76, 74, 72, 69][k % 8]; tone(stems['vocals'], t0, beat * 0.9, hz(mel), 0.1, 3, pan=-0.1)
x = sum(stems.values()); x = x / np.abs(x).max() * 0.8; scale = 0.8 / max(1e-9, np.abs(sum(stems.values())).max())
for k in stems: stems[k] *= scale
import soundfile as sf
sf.write(f'{OUT}/sep-mix.wav', x.T.astype(np.float32), SR, subtype='PCM_16')
wav = torch.tensor(np.round(x * 32767) / 32767, dtype=torch.float32)  # 16비트로 저장한 것과 같은 값
ref = wav.mean(0); m, sd = ref.mean(), ref.std(); inp = (wav - m) / sd
with torch.no_grad():
    est = apply_model(core, inp[None], shifts=0, split=True, overlap=0.25, progress=False)[0] * sd + m
def sdr(ref_, est_):
    num = np.sum(ref_ ** 2); den = np.sum((ref_ - est_) ** 2) + 1e-12; return 10 * np.log10(num / den + 1e-12)
res = {'sources': list(core.sources), 'sr': SR, 'len': n, 'segment': float(core.segment), 'sdr': {}, 'probe': {}}
for i, k in enumerate(core.sources):
    e = est[i].numpy(); res['sdr'][k] = round(float(sdr(stems[k], e)), 2)
    res['probe'][k] = {str(o): [round(float(v), 5) for v in e[0, o:o + 64]] for o in (44100, 300000, 400000)}
    res['probe'][k]['rms'] = round(float(np.sqrt(np.mean(e ** 2))), 6)
print('공식 경로 SDR (합성 정답 대비)', res['sdr'])
json.dump(res, open(f'{OUT}/sep-ref.json', 'w'))
meta = {fn: {'mb': round(os.path.getsize(f'{OUT}/{fn}') / 1048576, 2), 'sha256': hashlib.sha256(open(f'{OUT}/{fn}', 'rb').read()).hexdigest()} for fn in sorted(os.listdir(OUT)) if fn.endswith('.onnx')}
print(json.dumps(meta, indent=1)); json.dump(meta, open(f'{OUT}/sep-meta.json', 'w'), indent=1)
