# stdio 24장 V8-2d 받아쓰기 실측 (Actions · whisper-eval 워크플로) — 공용 코드는 whisper_common.py
#  진짜 사람 말(Google FLEURS 시험 문장 · CC BY 4.0) · tiny vs base · 한국어 CER · 영어 WER · 언어 · 문장 끝 부호 · 지어냄 · 시간. 오디오는 이 작업 안에서만 · 숫자만 남김
import os, sys, json, time, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from whisper_common import *
res = {'fleurs': 'google/fleurs (CC BY 4.0) test split', 'n': N, 'models': {'tiny': MC, 'base': BREV}, 'by': {}}
data = {'ko': fleurs('ko_kr'), 'en': fleurs('en_us')}
for mk, (enc, dec, h, dm) in MODELS.items():
    M = Model(enc, dec, h, dm)
    for lk, items in data.items():
        errs = tot = 0; lang_ok = ends = halluc = 0; secs = 0.0; dur = 0.0; ex = []; worst = []
        for x, ref in items:
            t0 = time.time(); r = M.run(x); secs += time.time() - t0; dur += len(x) / 16000
            hyp, rf = norm(r['text']), norm(ref)
            if lk == 'ko': a, b = hyp.replace(' ', ''), rf.replace(' ', '')
            else: a, b = hyp.split(), rf.split()
            errs += lev(a, b); tot += max(1, len(b)); lang_ok += r['lang'] == lk; ends += bool(re.search(r'[.?!。…]\s*$', r['text'])); halluc += (r['loop'] or r['silent'] or not r['text'])
            if len(ex) < 4: ex.append({'ref': ref[:80], 'hyp': r['text'][:80], 'lang': r['lang']})
            worst.append((lev(a, b) / max(1, len(b)), ref[:90], r['text'][:160], r['loop']))
        res['by'][f'{mk}-{lk}'] = {'err': round(errs / tot, 4), 'metric': 'CER' if lk == 'ko' else 'WER', 'langOk': round(lang_ok / len(items), 3), 'endPunct': round(ends / len(items), 3), 'empty': round(halluc / len(items), 3), 'rtf': round(secs / dur, 3), 'n': len(items), 'ex': ex, 'loops': sum(1 for w in worst if w[3]), 'worst': [{'e': round(w[0], 3), 'ref': w[1], 'hyp': w[2], 'loop': w[3]} for w in sorted(worst, reverse=True)[:5]]}
        print(mk, lk, json.dumps(res['by'][f'{mk}-{lk}'], ensure_ascii=False))
os.makedirs('models', exist_ok=True); json.dump(res, open('models/whisper-eval.json', 'w'), ensure_ascii=False, indent=1)
print('RESULT', json.dumps({k: {a: v[a] for a in ['err', 'metric', 'langOk', 'endPunct', 'empty', 'rtf']} for k, v in res['by'].items()}, ensure_ascii=False))

