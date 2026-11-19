#!/usr/bin/env node
// v1.0-C 장면 공장 — 파노라마 → 라이선스 검사 → 얼굴 · 번호판 흐림 → 큐브맵 6면 깊이 → 면 사이 (s, t) 최소제곱 정렬
//   → 지면 캘리브레이션((sin φ, d) RANSAC) → 2~3층 LDI → 가려진 곳 채우기 → 팩(WebP + 16비트 거리 PNG + manifest · sha256)
//
//  쓰는 법 (사람이 시작해요. 인터넷이 되는 곳 = GitHub Actions 의 build-scenes 워크플로):
//    node scripts/scene-factory.mjs --source synthetic --out public/packs/scenes            (이 저장소에 묶는 합성 시험 장면 3종)
//    node scripts/scene-factory.mjs --source polyhaven --ids "<hdri id> <hdri id>" --depth models/dav2s.onnx --face models/yunet.onnx --out out/scenes
//  필요: npm i --no-save sharp onnxruntime-node (워크플로가 설치). 깊이 모델이 없으면 polyhaven 은 멈춰요(합성 장면만 정답 흉내로 돌아요).
//  모델: Depth Anything V2 Small (Apache-2.0, onnx-community/depth-anything-v2-small) · YuNet (MIT, OpenCV Zoo) · LaMa(선택, --lama)
import { createHash } from 'node:crypto'
import { mkdir, writeFile, readFile, rename } from 'node:fs/promises'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { UA, fetchRetry } from '../src/scene/factory/sources.js'
import { SYNTH, renderSynth } from '../src/scene/factory/synth.js'
import { faceRGB, simDepthFaces, alignFaces, mergeFaces, FOV } from '../src/scene/factory/depth.js'
import { fitGroundPano, toMetric } from '../src/scene/factory/ground.js'
import { buildLDI, parallaxBudget } from '../src/scene/factory/ldi.js'
import { plateCandidates, plateKeep, yunetDecode, anonBlur, plateBlank, faceKeep, privacyLoad } from '../src/scene/factory/blur.js'
import { analyzePano, levelRotation, rotateEquirect } from '../src/scene/factory/level.js'
import { faceTiles, closeFace, mergeBoxes, normDisp, shotPlates } from '../src/scene/factory/shotkit.js'
import { checkLicense, attribution } from '../src/scene/license.js'
import { faceDir, uvOf } from '../src/scene/geom.js'

const A = Object.fromEntries(process.argv.slice(2).reduce((o, a, i, l) => (a.startsWith('--') ? [...o, [a.slice(2), l[i + 1] && !l[i + 1].startsWith('--') ? l[i + 1] : '1']] : o), []))
const OUT = A.out || 'out/scenes', SRC = A.source || 'synthetic', W = +(A.size || 2048), H = W / 2, FN = +(A.facepx || 518), SEED = +(A.seed || 7)
const sha = (b) => createHash('sha256').update(b).digest('hex')
const log = (...x) => console.log('[scene]', ...x)
let sharp = null; const img = async () => (sharp ||= (await import('sharp')).default)

// Poly Haven 고르기: 자산 목록(카테고리 · 태그) 점수로 주제마다 위에서 --count 개 (HDRI 하나는 점수가 가장 높은 주제 하나에만). 결정적 · 보고서에 기록
//  mood 는 앱의 장르 이름(gen/dna.js GNAME) · 분위기 이름(bright · dark · calm · energetic)과 같아야 장면 선택 점수가 맞아요
const THEMES = {
  city: { name: '노을 도시', want: { urban: 3, street: 3, road: 2, city: 2, sunset: 3, evening: 2, dusk: 2 }, avoid: { indoor: 5, studio: 5, night: 2 }, car: true, mood: { energy: 0.6, bpm: [85, 120], moods: ['bright', 'calm'], genres: ['citypop', 'lofi', 'disco', 'kpop'] } },
  nightcity: { name: '도시 야경', want: { urban: 3, city: 3, night: 4, lights: 2, street: 2 }, avoid: { indoor: 4, studio: 5, nature: 2 }, car: true, mood: { energy: 0.75, bpm: [88, 145], moods: ['dark', 'energetic'], genres: ['trap', 'boombap', 'house', 'techno', 'citypop'] } },
  daycity: { name: '한낮 도심', want: { urban: 3, city: 3, building: 2, buildings: 2, sunny: 2, clear: 2, midday: 2, day: 1 }, avoid: { night: 4, sunset: 2, indoor: 5, studio: 5 }, car: true, mood: { energy: 0.75, bpm: [100, 132], moods: ['bright', 'energetic'], genres: ['kpop', 'disco', 'rock', 'reggaeton'] } },
  market: { name: '겨울 · 축제 광장', want: { snow: 4, winter: 4, christmas: 4, market: 3, square: 2, night: 2 }, avoid: { studio: 2 }, mood: { energy: 0.45, bpm: [60, 120], moods: ['calm', 'bright'], genres: ['carol', 'waltz', 'ballad', 'lofi'] } },
  club: { name: '클럽 · 공연장', want: { neon: 4, stage: 3, concert: 3, hall: 3, 'artificial light': 3, indoor: 2, studio: 1, night: 1 }, avoid: { outdoor: 3, nature: 3 }, mood: { energy: 0.9, bpm: [118, 175], moods: ['energetic', 'dark'], genres: ['house', 'techno', 'dnb', 'trap', 'disco'] } },
  field: { name: '노을 들판', want: { sunset: 4, field: 3, meadow: 3, 'golden hour': 3, nature: 2, skies: 1 }, avoid: { urban: 3, indoor: 5, studio: 5 }, mood: { energy: 0.4, bpm: [60, 100], moods: ['calm', 'bright'], genres: ['ballad', 'lofi', 'waltz', 'citypop'] } },
  forest: { name: '숲', want: { forest: 4, trees: 3, woods: 3, nature: 2 }, avoid: { urban: 3, indoor: 5, studio: 5 }, mood: { energy: 0.35, bpm: [55, 95], moods: ['calm', 'dark'], genres: ['ballad', 'waltz', 'lofi'] } },
  coast: { name: '바다 · 해변', want: { beach: 4, sea: 4, coast: 3, ocean: 3, water: 2, sand: 2 }, avoid: { indoor: 5, studio: 5 }, mood: { energy: 0.65, bpm: [88, 125], moods: ['bright', 'energetic'], genres: ['reggaeton', 'citypop', 'disco', 'kpop'] } },
  industrial: { name: '공장 · 폐허', want: { industrial: 4, abandoned: 3, warehouse: 3, factory: 3, parking: 2, construction: 2 }, avoid: { nature: 2 }, mood: { energy: 0.8, bpm: [85, 150], moods: ['dark', 'energetic'], genres: ['techno', 'rock', 'boombap', 'trap', 'dnb'] } },
  cozy: { name: '아늑한 실내', want: { cozy: 4, room: 3, home: 3, living: 2, interior: 2, indoor: 2 }, avoid: { outdoor: 3, studio: 2 }, mood: { energy: 0.35, bpm: [60, 100], moods: ['calm', 'bright'], genres: ['lofi', 'ballad', 'waltz', 'carol'] } },
}
const COUNT = Math.max(1, +(A.count || 1)), ONLY = A.themes ? String(A.themes).split(/[\s,]+/).filter((t) => THEMES[t]) : Object.keys(THEMES)
async function pickPolyHaven() {
  const all = await (await fetch('https://api.polyhaven.com/assets?t=hdris')).json(), best = new Map()
  for (const [id, a] of Object.entries(all)) {
    const words = [...(a.categories || []), ...(a.tags || [])].map((x) => String(x).toLowerCase())
    for (const theme of ONLY) { const T = THEMES[theme]; let sc = 0; for (const w of words) { sc += T.want[w] || 0; sc -= T.avoid[w] || 0 } const b = best.get(id); if (sc > 0 && (!b || sc > b.score)) best.set(id, { id, theme, score: sc }) }
  }
  const out = []; for (const theme of ONLY) out.push(...[...best.values()].filter((x) => x.theme === theme).sort((x, y) => y.score - x.score || (x.id < y.id ? -1 : 1)).slice(0, COUNT))
  log('고른 HDRI:', out.map((x) => `${x.id}:${x.theme}(${x.score})`).join(' ')); return out
}
async function sources() {
  if (SRC === 'synthetic') return Object.entries(SYNTH).map(([id, s]) => ({ id, kind: 'synthetic', name: s.name, theme: id, mood: s.mood, car: s.car, license: 'CC0', author: 'AI 뮤직 DJ 스튜디오 장면 공장 (합성)', source: 'scripts/scene-factory.mjs', title: s.name + ' (합성 시험 장면)', make: s.make }))
  if (SRC === 'polyhaven') {
    const picks = A.ids && A.ids !== '1' ? String(A.ids).split(/[\s,]+/).filter(Boolean).map((x, i) => { const [id, theme] = x.split(':'); return { id, theme: THEMES[theme] ? theme : ONLY[i % ONLY.length] } }) : await pickPolyHaven(), out = []
    for (const p of picks) {
      const res = await fetch('https://api.polyhaven.com/info/' + p.id); if (!res.ok) { log('없는 HDRI (건너뜀):', p.id, res.status); continue }
      const info = await res.json(), T = THEMES[p.theme] /* Poly Haven 은 모든 자산이 CC0 */
      out.push({ id: p.id, kind: 'photo', name: `${T.name} · ${info.name || p.id}`, theme: p.theme, h: 1.6, mood: T.mood, car: !!T.car, license: 'CC0', author: Object.keys(info.authors || {}).join(', ') || 'Poly Haven', source: 'https://polyhaven.com/a/' + p.id, title: info.name || p.id, url: `https://dl.polyhaven.org/file/ph-assets/HDRIs/extra/Tonemapped%20JPG/${p.id}.jpg`, pick: p })
    }
    return out
  }
  throw new Error('모르는 --source: ' + SRC + ' (synthetic | polyhaven)')
}
async function loadRGB(s) {
  if (s.kind === 'synthetic') { const o = renderSynth(s.make(SEED), W, H); return { rgb: o.rgb, truth: o.r, h: o.h } }
  let buf; if (/^(file:|\/)/.test(s.url)) buf = await readFile(s.url.replace(/^file:\/\//, '')); else buf = Buffer.from(await (await fetchRetry(s.url)).arrayBuffer()) /* 로컬 파일은 오프라인 시험용 */
  const { data } = await (await img())(buf).resize(W, H, { fit: 'fill' }).removeAlpha().raw().toBuffer({ resolveWithObject: true })
  return { rgb: new Uint8Array(data), truth: null, h: s.h || 1.6 }
}
// 얼굴(YuNet) · 번호판(후보) 찾기 → 등장방형 상자. 합성 장면은 사람 · 번호판이 없어서 건너뛰어요(보고서에 그렇게 적어요)
// 번호판: 읽을 수 있는 것만 흐려요 — 수평선 아래 35° ~ 위 5° (땅 근처) · 거리 8m 안 (2048 파노라마에서 그보다 먼 번호판은 몇 화소라 읽을 수 없어요).
//  첫 실제 장면에서 휴리스틱 후보 159곳을 다 흐렸더니 창문 · 유리 격자가 번졌어요 → 이렇게 걸러요
async function privacy(s, rgb) {
  if (s.kind === 'synthetic') return { rgb, faces: 0, plates: 0, note: '합성 장면: 사람 · 번호판 없음 (생성)' }
  const boxes = [], d = [0, 0, 0], uv = [0, 0]; let faces = 0
  const toEq = (i, b, N) => { /* 면 상자 네 귀퉁이 → 등장방형 상자 */ const xs = [], ys = []; for (const [px, py] of [[b.x, b.y], [b.x + b.w, b.y], [b.x, b.y + b.h], [b.x + b.w, b.y + b.h]]) { faceDir(i, px, py, N, FOV, d); uvOf(d[0], d[1], d[2], uv); xs.push(uv[0] * W); ys.push(uv[1] * H) } let x0 = Math.min(...xs), x1 = Math.max(...xs); if (x1 - x0 > W / 2) { x0 = Math.max(...xs.filter((x) => x < W / 2)); x1 = Math.min(...xs.filter((x) => x >= W / 2)) + W; [x0, x1] = [x1 - W, x0] } return { x: x0, y: Math.min(...ys), w: x1 - x0, h: Math.max(...ys) - Math.min(...ys) } }
  if (A.face) {
    const ort = await import('onnxruntime-node'), sess = await ort.InferenceSession.create(A.face), S = 640
    for (let i = 0; i < 6; i++) { const f = faceRGB(rgb, W, H, i, S), t = new Float32Array(3 * S * S); for (let p = 0; p < S * S; p++) for (let c = 0; c < 3; c++) t[c * S * S + p] = f[p * 3 + (2 - c)] /* BGR */; const out = await sess.run({ [sess.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, S, S]) }); for (const b of yunetDecode(out, S, { thr: 0.7 })) { boxes.push(toEq(i, { x: b.x - b.w * 0.075, y: b.y - b.h * 0.075, w: b.w * 1.15, h: b.h * 1.15 }, S)); faces++ } /* 상자는 얼굴에 딱 맞게 (예전 1.4배는 주변까지 뭉갰어요) · 문턱 0.7 */ }
  } else if (s.kind !== 'synthetic') log('경고: --face 모델이 없어 얼굴을 찾지 못했어요 → 이 장면은 사람 확인 목록에 올라가요')
  const N = 512, plates = []; if (!s.blurred) for (let i = 0; i < 6; i++) if (i !== 2) for (const b of plateCandidates(faceRGB(rgb, W, H, i, N), N, N)) plates.push({ ...toEq(i, b, N), score: b.score })
  return { rgb, boxes, faces, cands: plates, plates: 0, review: !A.face, note: '' } /* 찾기만 해요: 깊이를 안 뒤 실제 크기로 거르고 흐려요 (build) */
}
// 깊이: 실제 모델(Depth Anything V2 Small) 또는 합성 장면의 정답 흉내
async function depthFaces(s, rgb, truth) {
  if (A.depth) {
    const ort = await import('onnxruntime-node'), sess = await ort.InferenceSession.create(A.depth), faces = [], M = [0.485, 0.456, 0.406], SD = [0.229, 0.224, 0.225]
    for (let i = 0; i < 6; i++) { const f = faceRGB(rgb, W, H, i, FN), t = new Float32Array(3 * FN * FN); for (let p = 0; p < FN * FN; p++) for (let c = 0; c < 3; c++) t[c * FN * FN + p] = (f[p * 3 + c] / 255 - M[c]) / SD[c]; const o = await sess.run({ [sess.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, FN, FN]) }); faces.push(Float32Array.from(o[sess.outputNames[0]].data)) }
    return { faces, how: 'Depth Anything V2 Small (ONNX · CPU)' }
  }
  if (!truth && A['fake-depth']) { const Wd = W, Hd = H, r = new Float32Array(Wd * Hd); for (let y = 0; y < Hd; y++) { const lat = (0.5 - (y + 0.5) / Hd) * Math.PI; for (let x = 0; x < Wd; x++) r[y * Wd + x] = lat < -0.05 ? Math.min(30, 1.6 / Math.sin(-lat)) : lat < 0.35 ? 30 : Infinity } return { faces: simDepthFaces(r, Wd, Hd, FN, { seed: SEED, noise: 0.01 }).faces, how: '시험 전용 가짜 깊이 (지면 + 30m 벽) — 실제 장면에 쓰지 않아요' } }
  if (!truth) throw new Error('깊이 모델(--depth)이 필요해요: ' + s.id)
  return { faces: simDepthFaces(truth, W, H, FN, { seed: SEED + 11, noise: 0.01 }).faces, how: '합성: 정답 거리에 면마다 다른 아핀 왜곡 + 잡음 1% (깊이 모델 흉내)' }
}
const DS = +(A.distDown || 4) /* 거리는 가로세로 1/4 (메쉬 꼭짓점이 최대 256×128 이라 충분) */
async function encodeDist(r0) { /* 거리 → log 정규화 16비트 (R = 위 8비트, G = 아래 8비트). 층 안에서 거리는 이미 매끈해서 log 평균으로 줄여요 */
  const w = W / DS, hh = H / DS, r = new Float32Array(w * hh); for (let y = 0; y < hh; y++) for (let x = 0; x < w; x++) { let s = 0; for (let j = 0; j < DS; j++) for (let i = 0; i < DS; i++) s += Math.log(Math.min(1000, r0[(y * DS + j) * W + x * DS + i])); r[y * w + x] = Math.exp(s / (DS * DS)) }
  let lo = Infinity, hi = 0; for (const v of r) { if (v < lo) lo = v; if (v > hi) hi = v } lo = Math.max(0.3, lo); hi = Math.max(lo * 1.01, Math.min(1000, hi))
  const a = Math.log(lo), b = Math.log(hi), px = Buffer.alloc(r.length * 3); for (let i = 0; i < r.length; i++) { const q = Math.round(Math.max(0, Math.min(1, (Math.log(Math.min(hi, Math.max(lo, r[i]))) - a) / (b - a))) * 65535); px[i * 3] = q >> 8; px[i * 3 + 1] = q & 255 }
  return { buf: await (await img())(px, { raw: { width: w, height: hh, channels: 3 } }).png({ compressionLevel: 9 }).toBuffer(), rmin: +lo.toFixed(4), rmax: +hi.toFixed(3) }
}
async function build(s) {
  const t0 = Date.now(), lic = checkLicense(s); if (!lic.ok) { log('거절', s.id, lic.why); return null }
  let { rgb, truth, h } = await loadRGB(s), lev = null
  if (s.kind !== 'synthetic') { /* 수평 맞추기: 기운 360 카메라 → 수평선이 물결쳐요. 1024×512 사본으로 위쪽 소실점을 재고 원본 구를 돌려요 (level.js) */
    const sm = W === 1024 ? rgb : new Uint8Array(await (await img())(Buffer.from(rgb), { raw: { width: W, height: H, channels: 3 } }).resize(1024, 512, { fit: 'fill' }).raw().toBuffer())
    lev = analyzePano(sm, 1024, 512); if (lev.level) rgb = rotateEquirect(rgb, W, H, levelRotation(lev.v))
  }
  const pv = await privacy(s, rgb)
  const N = FN, { faces, how } = await depthFaces(s, rgb, truth), ab = alignFaces(faces, N), g = mergeFaces(faces, ab, W, H, N), cal = fitGroundPano(g, W, H, { h })
  if (!cal || cal.inl < 0.15) { log('지면을 찾지 못했어요', s.id, cal && +cal.inl.toFixed(3)); return null } /* 실제 AI 깊이는 합성보다 거칠어요: 0.15~0.3 은 "약한 지면"으로 살리고 앱에서 이동 폭을 줄여요 */
  const r = toMetric(g, cal)
  if (pv.boxes) { /* 사생활 2판: 실제 크기로 거른 얼굴만 얼굴 모양대로 알맞게 흐리고, 번호판은 빈 판으로. 흐릴 곳이 너무 많으면 장면을 빼요 */
    const dq = s.desc ? (await import('../src/scene/descriptors.js')).unpack(s.desc) : null, IX = dq && (await import('../src/scene/descriptors.js')).IDX
    const indoor = dq ? ['place:room', 'place:cafe', 'place:club', 'place:hall', 'place:temple'].reduce((a, k) => a + (dq[IX[k]] || 0), 0) > 0.5 : false /* 5회차: 실내 스튜디오 · 작업장 무늬에 빈 판이 8곳씩 → 실내(방 · 카페 · 클럽 · 홀 · 성당)는 번호판을 찾지 않아요 (주차장 · 터널은 찾아요) */
    const fk = faceKeep(pv.boxes, r, W, H), pk = indoor ? [] : plateKeep(pv.cands || [], r, W, H, { h }), load = privacyLoad([...fk, ...pk], W, H)
    if (load.frac > 0.02 || load.n > 20) { log('흐릴 곳이 너무 많아 뺐어요', s.id, JSON.stringify({ frac: +load.frac.toFixed(4), n: load.n })); return null }
    rgb = plateBlank(anonBlur(rgb, W, H, fk), W, H, pk); pv.faces = fk.length; pv.plates = pk.length; pv.note = `얼굴 후보 ${pv.boxes.length}곳 중 실제 크기가 맞는 ${fk.length}곳을 알맞게 흐림 · ` + (indoor ? '실내라 번호판은 찾지 않음' : `번호판 후보 ${(pv.cands || []).length}곳 중 ${pk.length}곳은 빈 판으로`); pv.load = +load.frac.toFixed(4)
  }
  const rep = { id: s.id, depth: how, ground: { inliers: +cal.inl.toFixed(3), S: +cal.S.toFixed(5), T: +cal.T.toFixed(5), h }, privacy: { faces: pv.faces, plates: pv.plates, review: !!pv.review, note: pv.note, load: pv.load }, level: lev && { tilt: lev.tilt, conf: lev.conf, applied: lev.level, manh: lev.manh } }
  if (truth) { const e = []; for (let i = 0; i < r.length; i += 7) if (Number.isFinite(truth[i]) && truth[i] < 60) e.push(Math.abs(r[i] / truth[i] - 1)); e.sort((x, y) => x - y); rep.calErr = { median: +e[e.length >> 1].toFixed(4), p90: +e[(e.length * 0.9) | 0].toFixed(4) } }
  const L = buildLDI(rgb, r, W, H, { layers: +(A.layers || 3) }), _pb = parallaxBudget(L.layers, W, H), dir = join(OUT, s.id); await mkdir(dir, { recursive: true }); const files = {}, layers = []
  for (let k = 0; k < L.layers.length; k++) {
    const c = await (await img())(Buffer.from(L.layers[k].rgba), { raw: { width: W, height: H, channels: 4 } }).webp({ quality: +(A.q || 82), alphaQuality: 90, effort: 6 }).toBuffer(), dd = await encodeDist(L.layers[k].r)
    const cn = `l${k}.webp`, dn = `l${k}-dist.png`; await writeFile(join(dir, cn), c); await writeFile(join(dir, dn), dd.buf); files[cn] = sha(c); files[dn] = sha(dd.buf); layers.push({ color: cn, dist: dn, rmin: dd.rmin, rmax: dd.rmax })
  }
  const man = { format: 'djs-scene/1', id: s.id, name: s.name, kind: s.kind, label: s.kind === 'synthetic' ? '합성 시험 장면 · 실사 아님' : s.kind === 'reconstructed' ? '재구성 장면 · CC 자산' : '실제 장소 사진 · 깊이는 AI 추정', ...(s.desc ? { desc: s.desc, stats: s.stats } : {}), theme: s.theme, size: [W, H], distSize: [W / DS, H / DS], h, ground: { S: cal.S, T: cal.T, inl: cal.inl, ...(cal.inl < 0.3 ? { weak: true } : {}) }, maxOff: _pb.t /* 시차 예산 (m) */, ...(lev ? { level: { tilt: lev.tilt, applied: lev.level } } : {}), cuts: L.cuts, layers, mood: s.mood, car: s.car ? { ahead: 9, side: s.theme === 'city' ? -1.8 : 0 } : null, sources: [{ title: s.title, author: s.author, license: lic.L, source: s.source, text: attribution(s) }], review: !!pv.review, files }
  await writeFile(join(dir, 'manifest.json'), JSON.stringify(man, null, 1)); rep.maxOff = man.maxOff; rep.ms = Date.now() - t0; rep.bytes = Object.keys(files).length; log(s.id, JSON.stringify(rep))
  return { man, rep }
}
// --- 인서트 컷 (보통 사진: 멋진 차 · 사람들의 일상 · 곡 분위기에 맞는 순간) → djs-shot/1 ---
//  색 c.webp (가로 최대 1920) + 상대 깊이 d.png (8비트, 가로세로 1/2, 가까울수록 밝음). 앱은 깊이로 살짝 입체감 있게 다가가요(2.5D).
//  사생활: 얼굴은 원본 해상도 타일로 찾아 흐리고, 얼굴이 크게 나온 사진은 빼요. 차 번호판 후보는 크기로 걸러 흐려요.
async function buildShot(s) {
  const t0 = Date.now(), lic = checkLicense(s); if (!lic.ok) { log('거절', s.id, lic.why); return null }
  const buf = /^(file:|\/)/.test(s.url) ? await readFile(s.url.replace(/^file:\/\//, '')) : Buffer.from(await (await fetchRetry(s.url)).arrayBuffer()), sh = await img(), meta = await sh(buf).metadata()
  const SW = Math.min(1920, meta.width || 1920) & ~1, SH = Math.round(SW * (meta.height || SW / 1.6) / (meta.width || SW)) & ~1
  if (SW / SH < 1.3 || SW / SH > 2.4) { log('컷 거절 (가로 사진 아님)', s.id); return null }
  let rgb = new Uint8Array((await sh(buf).resize(SW, SH, { fit: 'fill' }).removeAlpha().raw().toBuffer({ resolveWithObject: true })).data)
  const faces = []
  if (A.face) {
    const ort = await import('onnxruntime-node'), sess = await ort.InferenceSession.create(A.face), S = 640
    for (const tl of faceTiles(SW, SH, S)) {
      const t = new Float32Array(3 * S * S), cw = Math.round(tl.w / tl.s), chh = Math.round(tl.h / tl.s)
      const tile = tl.s > 1 ? new Uint8Array((await sh(Buffer.from(rgb), { raw: { width: SW, height: SH, channels: 3 } }).resize(cw, chh, { fit: 'fill' }).raw().toBuffer())) : null
      for (let y = 0; y < Math.min(S, chh); y++) for (let x = 0; x < Math.min(S, cw); x++) { const i = tile ? (y * cw + x) * 3 : ((tl.y + y) * SW + tl.x + x) * 3, src = tile || rgb; for (let c = 0; c < 3; c++) t[c * S * S + y * S + x] = src[i + (2 - c)] /* BGR */ }
      const out = await sess.run({ [sess.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, S, S]) })
      for (const b of yunetDecode(out, S, { thr: 0.7 })) faces.push({ x: tl.x + (b.x - b.w * 0.075) * tl.s, y: tl.y + (b.y - b.h * 0.075) * tl.s, w: b.w * 1.15 * tl.s, h: b.h * 1.15 * tl.s, score: b.score })
    }
  } else log('경고: --face 모델이 없어 얼굴을 찾지 못했어요 → 이 컷은 사람 확인 목록에 올라가요')
  const fb = mergeBoxes(faces); if (closeFace(fb, SW, SH)) { log('컷 거절 (얼굴이 크게 나온 인물 사진)', s.id); return null }
  const pw = 1024, ph = Math.round(1024 * SH / SW), small = new Uint8Array(await sh(Buffer.from(rgb), { raw: { width: SW, height: SH, channels: 3 } }).resize(pw, ph, { fit: 'fill' }).raw().toBuffer()), k = SW / pw
  const plates = s.blurred ? [] : shotPlates(plateCandidates(small, pw, ph).map((b) => ({ x: b.x * k, y: b.y * k, w: b.w * k, h: b.h * k, score: b.score })), SW, SH, { sub: s.sub })
  const ld = fb.reduce((a, b) => a + b.w * b.h * Math.PI / 4, 0) / (SW * SH); if (ld > 0.03) { log('컷 거절 (흐릴 곳이 너무 많아요)', s.id, +ld.toFixed(4)); return null }
  rgb = plateBlank(anonBlur(rgb, SW, SH, fb, { wrap: false }), SW, SH, plates, { wrap: false }) /* 사생활 2판: 얼굴은 알맞게 흐리고 번호판은 빈 판으로 */
  let disp, how
  if (A.depth) {
    const ort = await import('onnxruntime-node'), sess = await ort.InferenceSession.create(A.depth), M = [0.485, 0.456, 0.406], SD = [0.229, 0.224, 0.225]
    const sq = new Uint8Array(await sh(Buffer.from(rgb), { raw: { width: SW, height: SH, channels: 3 } }).resize(FN, FN, { fit: 'fill' }).raw().toBuffer()), t = new Float32Array(3 * FN * FN)
    for (let p = 0; p < FN * FN; p++) for (let c = 0; c < 3; c++) t[c * FN * FN + p] = (sq[p * 3 + c] / 255 - M[c]) / SD[c]
    const o = await sess.run({ [sess.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, FN, FN]) }), raw = o[sess.outputNames[0]].data, n = Math.round(Math.sqrt(raw.length))
    disp = { d: normDisp(raw), w: n, h: n }; how = 'Depth Anything V2 Small (ONNX · CPU · 상대 깊이)'
  } else if (A['fake-depth']) { const n = 64, d = new Float32Array(n * n); for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) d[y * n + x] = y / (n - 1); disp = { d, w: n, h: n }; how = '시험 전용 가짜 깊이 (아래가 가까움) — 실제 컷에 쓰지 않아요' }
  else throw new Error('깊이 모델(--depth)이 필요해요: ' + s.id)
  const g8 = Buffer.alloc(disp.w * disp.h); for (let i = 0; i < g8.length; i++) g8[i] = Math.round(disp.d[i] * 255)
  const dpng = await sh(g8, { raw: { width: disp.w, height: disp.h, channels: 1 } }).resize(SW / 2, SH / 2, { fit: 'fill', kernel: 'cubic' }).png({ compressionLevel: 9 }).toBuffer()
  const cwebp = await sh(Buffer.from(rgb), { raw: { width: SW, height: SH, channels: 3 } }).webp({ quality: +(A.q || 84), effort: 6 }).toBuffer()
  const dir = join(OUT, s.id); await mkdir(dir, { recursive: true }); await writeFile(join(dir, 'c.webp'), cwebp); await writeFile(join(dir, 'd.png'), dpng)
  const man = { format: 'djs-shot/1', id: s.id, name: s.name, kind: 'shot', sub: s.sub, theme: s.sub, label: s.synthetic ? '합성 시험 컷 · 실사 아님' : '실제 사진 · 깊이는 AI 추정', ...(s.desc ? { desc: s.desc, stats: s.stats } : {}), size: [SW, SH], color: 'c.webp', disp: 'd.png', sources: [{ title: s.title, author: s.author, license: lic.L, source: s.source, text: attribution(s) }], privacy: { faces: fb.length, plates: plates.length }, review: !A.face && !s.synthetic, files: { 'c.webp': sha(cwebp), 'd.png': sha(dpng) } }
  await writeFile(join(dir, 'manifest.json'), JSON.stringify(man, null, 1)); const rep = { id: s.id, kind: 'shot', sub: s.sub, depth: how, privacy: man.privacy, size: man.size, ms: Date.now() - t0 }; log(s.id, JSON.stringify(rep))
  return { man, rep }
}
export { buildShot }
export { build, THEMES, loadRGB, img, sha }
const MAIN = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href /* 다른 스크립트(scene-mass)가 불러 쓸 때는 아래를 돌리지 않아요 */
if (MAIN) await (async () => {
const list = await sources(), done = []; await mkdir(OUT, { recursive: true })
for (const s of list) { try { const r = await build(s); if (r) done.push(r) } catch (e) { log('실패 (건너뜀):', s.id, String(e.message || e).slice(0, 200)) } }
// 가지 나누기 (--shard <MB>): jsDelivr 는 가지 하나 50MB 까지라, 장면을 scenes-1, scenes-2 … 가지에 나눠 담고 목록(index.json)만 scenes 가지에 둬요
const refOf = {}
if (A.shard) {
  const lim = +A.shard * 1048576; let k = 1, used = 0
  for (const { man } of done) { const sz = (await Promise.all(Object.keys(man.files).concat('manifest.json').map(async (f) => (await readFile(join(OUT, man.id, f))).length))).reduce((x, y) => x + y, 0); if (used && used + sz > lim) { k++; used = 0 } used += sz; refOf[man.id] = 'scenes-' + k }
  for (const { man } of done) { const to = join(OUT, refOf[man.id]); await mkdir(to, { recursive: true }); await rename(join(OUT, man.id), join(to, man.id)) }
  await mkdir(join(OUT, 'scenes'), { recursive: true })
}
const index = { format: 'djs-scenes/1', scenes: done.map(({ man }) => ({ id: man.id, name: man.name, kind: man.kind, label: man.label, theme: man.theme, mood: man.mood, review: man.review, ...(refOf[man.id] ? { ref: refOf[man.id] } : {}), manifest: man.id + '/manifest.json', sha: sha(JSON.stringify(man, null, 1)) })) }
const IX = A.shard ? join(OUT, 'scenes') : OUT
await writeFile(join(IX, 'index.json'), JSON.stringify(index, null, 1)); await writeFile(join(IX, 'report.json'), JSON.stringify(done.map((d) => d.rep), null, 1))
log('끝:', done.length, '/', list.length, '장면 →', OUT, A.shard ? '(가지 ' + new Set(Object.values(refOf)).size + '개)' : '')
})()
