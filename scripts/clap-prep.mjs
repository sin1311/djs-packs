// v1.1 ② CLAP 준비 (Actions · clap-prep 워크플로): 라이선스 · 파일 확인, 서술어 72개의 글 임베딩(앱에 실음), 시험 신호의 기준 특징 · 오디오 임베딩(앱 구현 확인용)
//  모델: Xenova/clap-htsat-unfused (laion/clap-htsat-unfused 의 ONNX 변환, Transformers.js 호환). 결과는 models/ 폴더 → models 가지.
import { writeFile, mkdir, readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { DESC, audioPrompt } from '../src/scene/descriptors.js'
const T = await import('@huggingface/transformers'), id = 'Xenova/clap-htsat-unfused', out = 'models', sha = (b) => createHash('sha256').update(b).digest('hex')
await mkdir(out, { recursive: true })
const J = async (u) => (await fetch(u)).json()
const card = await J('https://huggingface.co/api/models/laion/clap-htsat-unfused'), xc = await J('https://huggingface.co/api/models/' + id + '?blobs=true')
const pre = await J(`https://huggingface.co/${id}/resolve/main/preprocessor_config.json`)
const files = (xc.siblings || []).map((s) => ({ f: s.rfilename, mb: s.size ? +(s.size / 1048576).toFixed(1) : null }))
console.log('laion 라이선스', card.cardData && card.cardData.license, '| xenova 라이선스', xc.cardData && xc.cardData.license); console.log('파일', JSON.stringify(files)); console.log('전처리', JSON.stringify(pre))
// 글 임베딩 (72 서술어, 정규화 · int8)
const tok = await T.AutoTokenizer.from_pretrained(id), tm = await T.ClapTextModelWithProjection.from_pretrained(id), prompts = DESC.map(audioPrompt)
const te = (await tm(tok(prompts, { padding: true, truncation: true }))).text_embeds.tolist().map((v) => { const n = Math.hypot(...v); return v.map((x) => x / n) })
const q = te.map((v) => { const m = Math.max(...v.map(Math.abs)); return { s: +(m / 127).toFixed(8), q: v.map((x) => Math.round((x / m) * 127)) } })
await writeFile(out + '/clap-desc.json', JSON.stringify({ model: id, dim: te[0].length, prompts, int8: q }))
// 시험 신호 (결정적): ① 120BPM 킥 + 베이스 사인 ② 밝은 화음 + 하이햇 잡음. 48kHz · 10초
const sr = 48000, n = sr * 10, R = (() => { let a = 7; return () => { a = (a * 1664525 + 1013904223) >>> 0; return a / 4294967296 } })()
const sig = [new Float32Array(n), new Float32Array(n)]
for (let i = 0; i < n; i++) { const t = i / sr, b = t % 0.5, kick = Math.exp(-b * 18) * Math.sin(2 * Math.PI * (50 + 80 * Math.exp(-b * 30)) * b); sig[0][i] = 0.6 * kick + 0.25 * Math.sin(2 * Math.PI * 55 * t) * (0.5 + 0.5 * Math.sin(2 * Math.PI * 0.5 * t))
  const h = (t % 0.25) < 0.03 ? (R() * 2 - 1) * 0.15 : 0; sig[1][i] = 0.18 * (Math.sin(2 * Math.PI * 523.25 * t) + Math.sin(2 * Math.PI * 659.25 * t) + Math.sin(2 * Math.PI * 783.99 * t)) + h }
const fe = new T.ClapFeatureExtractor(pre), am = await T.ClapAudioModelWithProjection.from_pretrained(id, { dtype: 'q8' }), ref = []
for (const s of sig) { const { input_features } = await fe(s), e = (await am({ input_features })).audio_embeds.tolist()[0], nn = Math.hypot(...e)
  ref.push({ dims: input_features.dims, first: Array.from(input_features.data.slice(0, 64)).map((x) => +x.toFixed(4)), mean: +(input_features.data.reduce((a, b) => a + b, 0) / input_features.data.length).toFixed(4), emb: e.map((x) => +(x / nn).toFixed(5)) }) }
await writeFile(out + '/clap-ref.json', JSON.stringify({ sr, seconds: 10, signals: ['kick120+bass55', 'cmajor+hats'], ref }))
// 오디오 모델 파일 (q8) 을 같이 내보내요 (앱 거울: HF → jsDelivr(gh) → raw)
const qf = (files.find((f) => /onnx\/audio_model_quantized\.onnx$/.test(f.f)) || {}).f
if (qf) { const b = Buffer.from(await (await fetch(`https://huggingface.co/${id}/resolve/main/${qf}`)).arrayBuffer()); await writeFile(out + '/clap-audio-q.onnx', b); console.log('오디오 모델', qf, (b.length / 1048576).toFixed(1), 'MB', sha(b)) }
const meta = { model: id, upstream: 'laion/clap-htsat-unfused', license: { laion: card.cardData && card.cardData.license, xenova: xc.cardData && xc.cardData.license, code: 'CC0-1.0 (LAION-AI/CLAP)' }, files, pre, audioFile: qf, sha: {} }
for (const f of ['clap-desc.json', 'clap-ref.json', 'clap-audio-q.onnx']) { try { meta.sha[f] = sha(await readFile(out + '/' + f)) } catch (e) { /* 없음 */ } }
await writeFile(out + '/meta.json', JSON.stringify(meta, null, 1)); console.log('메타', JSON.stringify(meta.license), JSON.stringify(meta.sha))
