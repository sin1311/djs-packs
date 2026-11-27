# stdio 24장 V8-1 YAMNet 준비 (Actions · yamnet-prep 워크플로): 소리 사건 521가지(AudioSet 어휘) 모델을 ONNX 로.
#  원본: tensorflow/models research/audioset/yamnet (Apache-2.0 — 원문 licenses/audio/) · 가중치 storage.googleapis.com/audioset/yamnet.h5 (README 가 가리키는 곳).
#  앱은 소리 → 로그 멜 조각(96 × 64)을 직접 만들고(src/ai/yamnet.js) 이 ONNX 에는 조각만 넣어요(STFT 를 ONNX 에 안 넣음 — 브라우저 연산 호환).
#  그래서 앱 전처리가 TF 원본과 같은지 확인할 기준값(yamnet-ref.json)을 같이 만들어요: 결정적 시험 신호 5 + 말소리 1(flite) → 패딩 길이 · 조각 수 · 로그 멜 줄 · 점수.
import json, hashlib, os, sys, base64, urllib.request, subprocess
import numpy as np

OUT = 'models'; os.makedirs(OUT, exist_ok=True); os.makedirs('yam', exist_ok=True)
sha = lambda b: hashlib.sha256(b).hexdigest()
def get(u):
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={'user-agent': 'djs-packs-yamnet-prep'}), timeout=120) as r: return r.read()
        except Exception as e:
            print('다시', u, e)
    raise SystemExit('못 받음 ' + u)
api = json.loads(get('https://api.github.com/repos/tensorflow/models/commits/master'))
C = api['sha']; print('tensorflow/models 커밋', C)
base = f'https://raw.githubusercontent.com/tensorflow/models/{C}/research/audioset/yamnet/'
for f in ['params.py', 'yamnet.py', 'features.py', 'yamnet_class_map.csv', 'README.md']:
    open('yam/' + f, 'wb').write(get(base + f))
W = get('https://storage.googleapis.com/audioset/yamnet.h5'); open('yam/yamnet.h5', 'wb').write(W); print('가중치', len(W), sha(W))
sys.path.insert(0, 'yam')
import tensorflow as tf
import tf_keras  # yamnet.py 가 tf_keras(Keras 2) 를 씀 — 같은 것으로
import params as yparams, yamnet as ymodel, features as yfeat

p = yparams.Params()
print('params', {k: getattr(p, k) for k in ['sample_rate', 'stft_window_seconds', 'stft_hop_seconds', 'mel_bands', 'mel_min_hz', 'mel_max_hz', 'log_offset', 'patch_window_seconds', 'patch_hop_seconds', 'num_classes']})
full = ymodel.yamnet_frames_model(p); full.load_weights('yam/yamnet.h5')
inp = tf_keras.layers.Input(batch_shape=(None, 96, 64), name='patches')
pr, em = ymodel.yamnet(inp, p)
core = tf_keras.Model(inp, [pr, em]); core.load_weights('yam/yamnet.h5')

# 결정적 시험 신호(앱 시험이 같은 식으로 만듦 — LCG 는 siglip-prep 과 같음)
SR = 16000
def lcg(seed):
    a = seed
    def r():
        nonlocal a
        a = (a * 1664525 + 1013904223) % 4294967296
        return a / 4294967296
    return r
def sig(name):
    if name == 'sine': n = np.arange(32000); return 0.5 * np.sin(2 * np.pi * 1000 * n / SR)
    if name == 'chirp':
        t = np.arange(40000) / SR; T = 2.5; return 0.4 * np.sin(2 * np.pi * (200 * t + (4000 - 200) * t * t / (2 * T)))
    if name == 'noise': r = lcg(11); return np.array([0.3 * (r() * 2 - 1) for _ in range(19200)])
    if name == 'clicks':
        r = lcg(5); x = np.zeros(32000)
        for i in range(32000):
            t = i / SR; u = t % 0.3; nz = r() * 2 - 1
            if u < 0.12: x[i] = 0.9 * np.exp(-u * 30) * (np.sin(2 * np.pi * 70 * t) + 0.6 * nz)
        return x
    if name == 'short': n = np.arange(8000); return 0.5 * np.sin(2 * np.pi * 440 * n / SR)
SIG = {k: sig(k).astype(np.float32) for k in ['sine', 'chirp', 'noise', 'clicks', 'short']}
# 말소리(flite) — 16 kHz int16 로 기준값에 실음(앱 시험이 그대로 풀어 씀)
try:
    subprocess.run(['flite', '-t', 'Tonight, we drive into the city. This is where the night begins.', '-o', 'yam/sp.wav'], check=True)
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr0 = sf.read('yam/sp.wav', dtype='float32')
    if x.ndim > 1: x = x.mean(1)
    if sr0 != SR:
        from math import gcd
        g = gcd(SR, sr0); x = resample_poly(x, SR // g, sr0 // g).astype(np.float32)
    x = x[:SR * 3]; q = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16); SIG['speech'] = (q.astype(np.float32) / 32767)
    SPEECH_B64 = base64.b64encode(q.tobytes()).decode()
except Exception as e:
    print('말소리 없이', e); SPEECH_B64 = None

names = [l.split(',', 2)[2].strip().strip('"') for l in open('yam/yamnet_class_map.csv').read().strip().split('\n')[1:]]
mids = [l.split(',', 2)[1] for l in open('yam/yamnet_class_map.csv').read().strip().split('\n')[1:]]
assert len(names) == 521, len(names)
ref = {'commit': C, 'h5sha': sha(W), 'sr': SR, 'signals': {}}
for k, x in SIG.items():
    wpad = yfeat.pad_waveform(tf.constant(x), p)
    lm, patches = yfeat.waveform_to_log_mel_spectrogram_patches(wpad, p)
    lm = lm.numpy(); patches = patches.numpy()
    sc, _, _ = full(tf.constant(x)); sc = sc.numpy()
    sc2, _ = core(tf.constant(patches)); sc2 = sc2.numpy()
    d = float(np.abs(sc - sc2).max()); print(k, '조각', patches.shape, '원본 ↔ 조각 모델 차', d); assert d < 1e-4
    mean = sc.mean(0); top = np.argsort(-mean)[:5]
    F = lm.shape[0]; rows = sorted(set([0, F // 2, F - 1]))
    ref['signals'][k] = {'n': int(len(x)), 'padded': int(wpad.shape[0]), 'frames': int(F), 'patches': int(patches.shape[0]),
        'rows': {str(i): [round(float(v), 5) for v in lm[i]] for i in rows}, 'frameSum': [round(float(v), 3) for v in lm.sum(1)],
        'top': [[int(i), names[i], round(float(mean[i]), 5)] for i in top], 'p0': [round(float(v), 6) for v in sc[0]], 'top1': [int(np.argmax(r)) for r in sc]}
    print(k, ref['signals'][k]['top'])
if SPEECH_B64: ref['speechB64'] = SPEECH_B64

import tf2onnx, onnx, onnxruntime as ort
spec = (tf.TensorSpec([None, 96, 64], tf.float32, name='patches'),)
mp, _ = tf2onnx.convert.from_keras(core, input_signature=spec, opset=13, output_path=OUT + '/yamnet.onnx')
s = ort.InferenceSession(OUT + '/yamnet.onnx'); ins = [i.name for i in s.get_inputs()]; outs = [o.name for o in s.get_outputs()]; print('ONNX 입출력', ins, outs, [o.shape for o in s.get_outputs()])
worst = 0
for k, x in SIG.items():
    wpad = yfeat.pad_waveform(tf.constant(x), p); _, patches = yfeat.waveform_to_log_mel_spectrogram_patches(wpad, p); patches = patches.numpy()
    o = s.run(None, {ins[0]: patches}); sc2, _ = core(tf.constant(patches))
    oi = [i for i, sh in enumerate(s.get_outputs()) if sh.shape[-1] == 521][0]
    worst = max(worst, float(np.abs(o[oi] - sc2.numpy()).max()))
print('ONNX ↔ TF 차', worst); assert worst < 1e-4
ref['io'] = {'in': ins[0], 'out': [o.name for o in s.get_outputs() if o.shape[-1] == 521][0], 'emb': [o.name for o in s.get_outputs() if o.shape[-1] == 1024][0]}
b = open(OUT + '/yamnet.onnx', 'rb').read(); ref['onnx'] = {'bytes': len(b), 'sha': sha(b)}
json.dump(ref, open(OUT + '/yamnet-ref.json', 'w'))
json.dump({'source': f'tensorflow/models@{C} research/audioset/yamnet/yamnet_class_map.csv', 'names': names, 'mids': mids}, open(OUT + '/yamnet-classes.json', 'w'), ensure_ascii=False)
for f in ['yamnet.onnx', 'yamnet-ref.json', 'yamnet-classes.json']:
    bb = open(OUT + '/' + f, 'rb').read(); print(f, len(bb), sha(bb))
