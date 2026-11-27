// stdio 24장 V3-2b-2 SigLIP 준비 (Actions · siglip-prep 워크플로): 상황 어휘 문장(data/situ-prompts.json)의 글 임베딩(앱에 실음) · 점수 눈금(logit 기울기 · 절편) ·
//  시험 사진 3장의 사진 임베딩(앱 구현 확인용) · 사진 쪽 int8 ONNX → models/ 폴더(워크플로가 models 가지에 더함).
//  모델: Xenova/siglip-base-patch16-224 (google/siglip-base-patch16-224 의 ONNX 변환 · apache-2.0 — 원문 licenses/vision/).
import { writeFile, mkdir, readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
const T = await import('@huggingface/transformers'), id = 'Xenova/siglip-base-patch16-224', up = 'google/siglip-base-patch16-224', out = 'models', sha = (b) => createHash('sha256').update(b).digest('hex')
await mkdir(out, { recursive: true })
const J = async (u) => (await fetch(u)).json()
const card = await J('https://huggingface.co/api/models/' + up), xc = await J('https://huggingface.co/api/models/' + id + '?blobs=true')
const pre = await J(`https://huggingface.co/${id}/resolve/main/preprocessor_config.json`)
const files = (xc.siblings || []).map((s) => ({ f: s.rfilename, mb: s.size ? +(s.size / 1048576).toFixed(1) : null }))
console.log('라이선스', card.cardData && card.cardData.license, '| 파일', JSON.stringify(files.filter((x) => /onnx/.test(x.f)))); console.log('전처리', JSON.stringify(pre))
const P = JSON.parse(await readFile('data/situ-prompts.json', 'utf8')), prompts = P.items.map((x) => `This is a photo of ${x.en}.`)
/* 글 임베딩(정규화 · int8) */
const tok = await T.AutoTokenizer.from_pretrained(id), tm = await T.SiglipTextModel.from_pretrained(id)
const ti = tok(prompts, { padding: 'max_length', truncation: true }), tOut = await tm(ti), te = tOut.pooler_output.tolist().map((v) => { const n = Math.hypot(...v); return v.map((x) => x / n) })
const q = te.map((v) => { const m = Math.max(...v.map(Math.abs)); return { s: +(m / 127).toFixed(8), q: v.map((x) => Math.round((x / m) * 127)) } })
/* 시험 사진(결정적 · 224×224): 바닷가 · 눈 · 밤 도시 불빛 */
const R = (() => { let a = 7; return () => { a = (a * 1664525 + 1013904223) >>> 0; return a / 4294967296 } })()
const mk = (f) => { const d = new Uint8ClampedArray(224 * 224 * 3); for (let y = 0; y < 224; y++) for (let x = 0; x < 224; x++) { const c = f(x, y), i = (y * 224 + x) * 3; d[i] = c[0]; d[i + 1] = c[1]; d[i + 2] = c[2] } return new T.RawImage(d, 224, 224, 3) }
const imgs = { beach: mk((x, y) => (y < 75 ? [90, 160, 235] : y < 150 ? [40, 110, 170] : [225, 200, 150])), snow: mk((x, y) => (y < 60 ? [200, 210, 225] : [240 + (R() * 10 | 0), 242, 246])), citynight: mk(() => (R() < 0.05 ? [255, 220, 120] : R() < 0.02 ? [255, 60, 160] : [12, 12, 24])) }
const proc = await T.AutoProcessor.from_pretrained(id), vq = await T.SiglipVisionModel.from_pretrained(id, { dtype: 'q8' }), vf = await T.SiglipVisionModel.from_pretrained(id)
const full = await T.SiglipModel.from_pretrained(id), ref = [], xs = [], ys = []
for (const [name, img] of Object.entries(imgs)) {
  const { pixel_values } = await proc(img), eq = (await vq({ pixel_values })).pooler_output.tolist()[0], ef = (await vf({ pixel_values })).pooler_output.tolist()[0], nq = Math.hypot(...eq), nf = Math.hypot(...ef)
  const cos = te.map((t) => t.reduce((a, v, k) => a + v * ef[k] / nf, 0)), cq = te.map((t) => t.reduce((a, v, k) => a + v * eq[k] / nq, 0))
  const fo = await full({ ...ti, pixel_values }), lg = fo.logits_per_image.tolist()[0]; for (let k = 0; k < lg.length; k++) { xs.push(cos[k]); ys.push(lg[k]) }
  const top = (c) => c.map((v, k) => [P.items[k].id, v]).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([i, v]) => [i, +v.toFixed(4)])
  ref.push({ name, dims: pixel_values.dims, first: Array.from(pixel_values.data.slice(0, 32)).map((x) => +x.toFixed(4)), emb: eq.map((x) => +(x / nq).toFixed(5)), topQ: top(cq), topF: top(cos), agreeTop5: top(cq).filter(([i]) => top(cos).some(([j]) => j === i)).length })
  console.log('시험', name, 'q8', JSON.stringify(top(cq)), '| fp32', JSON.stringify(top(cos)))
}
/* 점수 눈금: logits = a · cos + b (최소 제곱) — 앱은 sigmoid(a · cos + b) 로 상황마다 확률 */
const n = xs.length, mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n, a = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0) / xs.reduce((s, x) => s + (x - mx) ** 2, 0), b = my - a * mx
const resid = Math.max(...xs.map((x, i) => Math.abs(a * x + b - ys[i])))
console.log('눈금', a.toFixed(4), b.toFixed(4), '최대 오차', resid.toFixed(5))
await writeFile(out + '/siglip-situ.json', JSON.stringify({ model: id, upstream: up, dim: te[0].length, template: 'This is a photo of {en}.', ids: P.items.map((x) => x.id), int8: q, scale: +a.toFixed(6), bias: +b.toFixed(6), resid: +resid.toFixed(6), pre: { size: pre.size, mean: pre.image_mean, std: pre.image_std, rescale: pre.rescale_factor, resample: pre.resample } }))
await writeFile(out + '/siglip-ref.json', JSON.stringify({ ref }))
/* 사진 쪽 int8 ONNX — 45MB 넘으면 조각(raw.githubusercontent 100MB 한도 · 앱 models.js parts) */
const vqf = (files.find((f) => /onnx\/vision_model_quantized\.onnx$/.test(f.f)) || {}).f, meta = { model: id, upstream: up, license: { google: card.cardData && card.cardData.license, xenova: xc.cardData && xc.cardData.license }, files: files.filter((x) => /onnx|json/.test(x.f)), visionFile: vqf, parts: [], sha: {} }
if (vqf) { const buf = Buffer.from(await (await fetch(`https://huggingface.co/${id}/resolve/main/${vqf}`)).arrayBuffer()), PART = 45 * 1048576; meta.size = buf.length; meta.sha.whole = sha(buf)
  if (buf.length <= PART) { await writeFile(out + '/siglip-vision-q.onnx', buf); meta.parts.push({ f: 'siglip-vision-q.onnx', sha: sha(buf), size: buf.length }) }
  else for (let i = 0, k = 0; i < buf.length; i += PART, k++) { const p = buf.subarray(i, i + PART), f = `siglip-vision-q.part${k}`; await writeFile(out + '/' + f, p); meta.parts.push({ f, sha: sha(p), size: p.length }) }
  console.log('사진 쪽 모델', vqf, (buf.length / 1048576).toFixed(1), 'MB', meta.parts.length, '조각') }
for (const f of ['siglip-situ.json', 'siglip-ref.json']) meta.sha[f] = sha(await readFile(out + '/' + f))
await writeFile(out + '/siglip-meta.json', JSON.stringify(meta, null, 1)); console.log('메타', JSON.stringify(meta.sha), JSON.stringify(meta.parts))
