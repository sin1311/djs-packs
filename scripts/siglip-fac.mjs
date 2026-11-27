// stdio 24장 V3-2c 상황 요인 어휘 굽기 (Actions · siglip-fac 워크플로): data/fac-prompts.json(축 × 문장) → SigLIP 글 임베딩 int8 + 시험 사진 3장의 축별 상위(앱 구현 확인용)
//  점수 눈금(logit 기울기 · 절편)은 모델 상수라 siglip-prep 에서 구한 값을 그대로 다시 구해 확인해요. 결과 models/siglip-fac.json · siglip-fac-ref.json → models 가지에 더함.
import { writeFile, mkdir, readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
const T = await import('@huggingface/transformers'), id = 'Xenova/siglip-base-patch16-224', out = 'models', sha = (b) => createHash('sha256').update(b).digest('hex')
await mkdir(out, { recursive: true })
const P = JSON.parse(await readFile('data/fac-prompts.json', 'utf8')), prompts = P.items.map((x) => x.en)
const tok = await T.AutoTokenizer.from_pretrained(id), tm = await T.SiglipTextModel.from_pretrained(id), ti = tok(prompts, { padding: 'max_length', truncation: true })
const te = (await tm(ti)).pooler_output.tolist().map((v) => { const n = Math.hypot(...v); return v.map((x) => x / n) })
const q = te.map((v) => { const m = Math.max(...v.map(Math.abs)); return { s: +(m / 127).toFixed(8), q: v.map((x) => Math.round((x / m) * 127)) } })
const R = (() => { let a = 7; return () => { a = (a * 1664525 + 1013904223) >>> 0; return a / 4294967296 } })()
const mk = (f) => { const d = new Uint8ClampedArray(224 * 224 * 3); for (let y = 0; y < 224; y++) for (let x = 0; x < 224; x++) { const c = f(x, y), i = (y * 224 + x) * 3; d[i] = c[0]; d[i + 1] = c[1]; d[i + 2] = c[2] } return new T.RawImage(d, 224, 224, 3) }
const imgs = { beach: mk((x, y) => (y < 75 ? [90, 160, 235] : y < 150 ? [40, 110, 170] : [225, 200, 150])), snow: mk((x, y) => (y < 60 ? [200, 210, 225] : [240 + (R() * 10 | 0), 242, 246])), citynight: mk(() => (R() < 0.05 ? [255, 220, 120] : R() < 0.02 ? [255, 60, 160] : [12, 12, 24])) }
const proc = await T.AutoProcessor.from_pretrained(id), vq = await T.SiglipVisionModel.from_pretrained(id, { dtype: 'q8' }), full = await T.SiglipModel.from_pretrained(id), xs = [], ys = [], ref = []
for (const [name, img] of Object.entries(imgs)) {
  const { pixel_values } = await proc(img), e = (await vq({ pixel_values })).pooler_output.tolist()[0], n = Math.hypot(...e), cq = te.map((t) => t.reduce((a, v, k) => a + v * e[k] / n, 0))
  const fo = await full({ ...ti, pixel_values }), lg = fo.logits_per_image.tolist()[0], ef = fo.image_embeds.tolist()[0], nf = Math.hypot(...ef); te.forEach((t, k) => { xs.push(t.reduce((a, v, j) => a + v * ef[j] / nf, 0)); ys.push(lg[k]) })
  const byAxis = {}; P.items.forEach((it, k) => { (byAxis[it.axis] = byAxis[it.axis] || []).push([it.id, +cq[k].toFixed(4)]) }); for (const a in byAxis) byAxis[a] = byAxis[a].sort((u, v) => v[1] - u[1]).slice(0, 3)
  ref.push({ name, top: byAxis }); console.log('시험', name, JSON.stringify(byAxis))
}
const N = xs.length, mx = xs.reduce((a, b) => a + b, 0) / N, my = ys.reduce((a, b) => a + b, 0) / N, a = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0) / xs.reduce((s, x) => s + (x - mx) ** 2, 0), b = my - a * mx
console.log('눈금', a.toFixed(4), b.toFixed(4))
await writeFile(out + '/siglip-fac.json', JSON.stringify({ model: id, dim: te[0].length, ids: P.items.map((x) => x.id), int8: q, scale: +a.toFixed(6), bias: +b.toFixed(6) }))
await writeFile(out + '/siglip-fac-ref.json', JSON.stringify({ ref }))
for (const f of ['siglip-fac.json', 'siglip-fac-ref.json']) console.log('sha', f, sha(await readFile(out + '/' + f)))
