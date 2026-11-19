#!/usr/bin/env node
// v1.0-D 대량 장면 공장 — 후보 5000곳 이상 → 품질 관문 → 분위기 칸 채우기로 2000~3000곳 → 나눠 만들기 → 합치기 (+ 하늘 도서관)
//  단계 (GitHub Actions 의 build-mass 워크플로가 차례로 불러요. 사람이 시작):
//    --stage catalog  [--target 5000] [--mapillary <토큰>] [--terms "…"]   → work/candidates.json
//    --stage gate     [--clip]                                              → work/scored.json   (작은 사본 1024×512 · 품질 · 색 통계 · 분위기 서술어)
//    --stage curate   [--total 2500]                                        → work/selected.json · work/gaps.json · work/skies.json
//    --stage build    --shard i/n --out out/shard-i [--depth … --face …]   → 장면 팩 + out/shard-i/part.json
//    --stage merge    --parts "out/shard-*/part.json" [--base-hf <사용자/데이터셋>]  → out/scenes/index.json
//    --stage skies    --out out/skies                                       → 하늘 도서관 (sky-<id>.webp + SH 9계수)
//  모델: CLIP ViT-B/32 (MIT, Xenova/clip-vit-base-patch32 · @huggingface/transformers) — 없으면 색 통계 + 출처 글자로 서술어를 정해요.
import { mkdir, writeFile, readFile, readdir, stat } from 'node:fs/promises'
import { join, dirname } from 'node:path'
import { SOURCES, UA, isPano, fetchRetry } from '../src/scene/factory/sources.js'
import { gate } from '../src/scene/factory/quality.js'
import { dedupe, curate, gapTerms } from '../src/scene/factory/curate.js'
import { checkLicense } from '../src/scene/license.js'
import { DESC, N, GROUPS, groupOf, prompt, sceneDist, pack, topDesc, IDX } from '../src/scene/descriptors.js'
import { shProject, lin } from '../src/scene/sh.js'
import { faceRGB } from '../src/scene/factory/depth.js'

const A = Object.fromEntries(process.argv.slice(2).reduce((o, a, i, l) => (a.startsWith('--') ? [...o, [a.slice(2), l[i + 1] && !l[i + 1].startsWith('--') ? l[i + 1] : '1']] : o), []))
const WORK = A.work || 'work', log = (...x) => console.log('[mass]', ...x)
const rd = async (f, d) => { try { return JSON.parse(await readFile(f, 'utf8')) } catch (e) { return d } }
const wr = async (f, o) => { await mkdir(dirname(f), { recursive: true }); await writeFile(f, JSON.stringify(o)) }
let sharp = null; const img = async () => (sharp ||= (await import('sharp')).default)
const get = async (url) => { if (/^(file:|\/)/.test(url)) return readFile(url.replace(/^file:\/\//, '')); const r = await fetchRetry(url); return Buffer.from(await r.arrayBuffer()) } /* 로컬 파일은 오프라인 시험용 */
const pool = async (items, n, fn) => { let k = 0; await Promise.all(Array.from({ length: n }, async () => { while (k < items.length) { const i = k++; try { await fn(items[i], i) } catch (e) { items[i]._err = String(e.message || e).slice(0, 120) } } })) }

// 출처 글자(제목 · 태그 · 설명) → 서술어 점수 (CLIP 없을 때 · 있을 때도 약하게 섞어요)
const KW = { 'place:street': /street|road|avenue|boulevard|rue|calle|strasse|거리/i, 'place:alley': /alley|lane|narrow|골목/i, 'place:downtown': /downtown|skyscraper|skyline|tower|financial/i, 'place:highway': /highway|motorway|freeway|road/i, 'place:harbor': /harbou?r|port|pier|dock|marina/i, 'place:beach': /beach|coast|seaside|shore|sand/i, 'place:lake': /lake|river|canal|pond|water/i, 'place:forest': /forest|wood|trees|jungle/i, 'place:mountain': /mountain|alps|peak|hill|valley/i, 'place:field': /field|meadow|farm|grass|prairie/i, 'place:desert': /desert|dune|canyon/i, 'place:plaza': /square|plaza|piazza|place\b/i, 'place:market': /market|bazaar|stall|christmas/i, 'place:station': /station|railway|metro|subway|platform/i, 'place:rooftop': /rooftop|terrace|roof/i, 'place:park': /park|garden/i, 'place:oldtown': /old town|medieval|historic|castle|cathedral/i, 'place:temple': /temple|church|cathedral|chapel|mosque|shrine/i, 'place:industrial': /industrial|factory|warehouse|plant|hangar/i, 'place:parking': /parking|garage/i, 'place:tunnel': /tunnel|underground|cave/i, 'place:club': /club|disco|dance/i, 'place:hall': /hall|theat(er|re)|auditorium|concert|opera/i, 'place:cafe': /cafe|café|bar|pub|restaurant/i, 'place:room': /room|interior|indoor|apartment|lobby|studio/i, 'place:stadium': /stadium|arena/i, 'place:amusement': /amusement|fair|carousel|ferris/i,
  'time:night': /night|nocturn|evening/i, 'time:sunset': /sunset|dusk/i, 'time:dawn': /dawn|sunrise|morning/i, 'time:noon': /noon|midday|sunny/i, 'weather:snow': /snow|winter/i, 'weather:fog': /fog|mist|haze/i, 'weather:rain': /rain|wet/i, 'weather:overcast': /overcast|cloudy|grey|gray/i, 'weather:storm': /storm/i, 'light:neon': /neon/i, 'light:strings': /christmas|fairy lights|string lights/i, 'light:lamps': /lamp|lantern/i, 'light:citylights': /city lights|illuminat/i }
const kwScores = (text) => { const sc = new Float32Array(N); for (const [k, re] of Object.entries(KW)) if (re.test(text || '')) sc[IDX[k]] += 1.2; return sc }
const THEME_OF = (q) => { const [a, b] = groupOf('place'); let bi = a; for (let i = a; i < b; i++) if (q[i] > q[bi]) bi = i; const p = DESC[bi].id, t = q[IDX['time:night']] + q[IDX['time:midnight']]; return /club|hall/.test(p) ? 'club' : /market|plaza/.test(p) ? 'market' : /forest|mountain/.test(p) ? 'forest' : /field|park|desert/.test(p) ? 'field' : /beach|harbor|lake/.test(p) ? 'coast' : /industrial|parking|tunnel/.test(p) ? 'industrial' : /room|cafe/.test(p) ? 'cozy' : t > 0.4 ? 'nightcity' : 'city' }

// CLIP (선택): 앞 · 뒤 · 좌 · 우 원근 4장 → 이미지 임베딩 평균 · 서술어 문장 임베딩 → 갈래마다 softmax(100·cos)
async function clipScorer() {
  try {
    const T = await import('@huggingface/transformers'), id = 'Xenova/clip-vit-base-patch32'
    const tok = await T.AutoTokenizer.from_pretrained(id), proc = await T.AutoProcessor.from_pretrained(id), tm = await T.CLIPTextModelWithProjection.from_pretrained(id), vm = await T.CLIPVisionModelWithProjection.from_pretrained(id)
    const te = (await tm(tok(DESC.map(prompt), { padding: true, truncation: true }))).text_embeds, tv = te.tolist().map((v) => { const n = Math.hypot(...v); return v.map((x) => x / n) })
    return async (rgb, W, H) => {
      const embs = []; for (const f of [0, 1, 4, 5]) { const face = faceRGB(rgb, W, H, f, 224), im = new T.RawImage(new Uint8ClampedArray(face), 224, 224, 3); const e = (await vm(await proc(im))).image_embeds.tolist()[0], n = Math.hypot(...e); embs.push(e.map((x) => x / n)) }
      const m = embs[0].map((_, k) => embs.reduce((s, e) => s + e[k], 0) / embs.length), sc = new Float32Array(N)
      for (const [g] of GROUPS) { const [a, b] = groupOf(g), lg = []; for (let i = a; i < b; i++) lg.push(100 * tv[i].reduce((s, x, k) => s + x * m[k], 0)); const mx = Math.max(...lg), ex = lg.map((x) => Math.exp(x - mx)), s = ex.reduce((x, y) => x + y, 0); for (let i = a; i < b; i++) sc[i] = ex[i - a] / s }
      return sc
    }
  } catch (e) { log('CLIP 을 못 불러와서 색 통계 + 글자로 서술어를 정해요:', String(e.message || e).slice(0, 120)); return null }
}

async function catalog() {
  if (A.fixture) { const fx = await rd(A.fixture, []); log('시험 후보 (오프라인):', fx.length); await wr(join(WORK, 'candidates.json'), fx.map((c) => ({ ...c, pack: checkLicense(c).pack, L: checkLicense(c).L })).filter((c) => c.pack)); return }
  const target = +(A.target || 5000), gapsPrev = A.gaps ? (await rd(A.gaps, { terms: [] })).terms || [] : [], terms = [...new Set([...gapsPrev, ...String(A.terms || 'city street|night city|old town|harbor|beach|forest|mountain|snow|market|square|station|park|industrial|interior|church|bridge|river|desert|rooftop|neon|festival').split('|')])], all = [] /* --gaps: 지난번 빈 칸 검색어를 앞에 */
  if (gapsPrev.length) log('지난번 빈 칸 검색어:', gapsPrev.join(' · '))
  const runs = [['polyhaven', {}], ['ambientcg', {}], ['wikimedia', { terms, per: Math.ceil(target / terms.length) }], ['panoramax', {}], ['mapillary', { token: A.mapillary || process.env.MAPILLARY_TOKEN }]]
  for (const [k, o] of runs) { try { const got = await SOURCES[k](fetch, o); log(k, got.length); all.push(...got) } catch (e) { log(k, '실패 (건너뜀):', String(e.message || e).slice(0, 160)) } }
  const seen = new Set(), ok = []; for (const c of all) { if (seen.has(c.id) || !c.url) continue; seen.add(c.id); const L = checkLicense(c); if (!L.ok) continue; c.pack = L.pack; c.L = L.L; ok.push(c) }
  log('후보:', all.length, '→ 라이선스 통과', ok.length, ok.length < target ? `(목표 ${target} 보다 적어요 — --terms 로 더 모아요)` : ''); await wr(join(WORK, 'candidates.json'), ok)
}
async function gateStage() {
  const cands = await rd(join(WORK, 'candidates.json'), []), clip = A.clip ? await clipScorer() : null, out = []
  await pool(cands, +(A.conc || 6), async (c) => {
    const buf = await get(c.thumb || c.url), { data, info } = await (await img())(buf).resize(1024, 512, { fit: 'fill' }).removeAlpha().raw().toBuffer({ resolveWithObject: true }), rgb = new Uint8Array(data)
    const g = gate(rgb, info.width, info.height, { w: c.w, h: c.h }, { minW: c.sky ? 2000 : 4000, sky: !!c.sky }); c.q = g.score; c.hash = g.hash; c.stats = g.stats; c.why = g.why; c.pass = g.pass
    const cs = clip ? await clip(rgb, info.width, info.height) : null, kw = kwScores(c.text); if (cs) for (let i = 0; i < N; i++) cs[i] += kw[i] * 0.15
    const d = sceneDist({ clip: cs || kw, stats: g.stats }); c.desc = pack(d); c.theme = THEME_OF(d)
    out.push(c)
  })
  log('관문:', out.length, '/', cands.length, '받음 · 통과', out.filter((c) => c.pass).length); await wr(join(WORK, 'scored.json'), out)
  const by = {}; for (const c of cands) { const b = (by[c.src] = by[c.src] || { n: 0, got: 0, pass: 0, err: {} }); b.n++; if (c._err) { const k = c._err.split(' ').slice(0, 2).join(' '); b.err[k] = (b.err[k] || 0) + 1 } else if (out.includes(c)) { b.got++; if (c.pass) b.pass++ } }
  for (const [k, b] of Object.entries(by)) log('관문 ·', k, `후보 ${b.n} · 받음 ${b.got} · 통과 ${b.pass}`, Object.keys(b.err).length ? '· 실패 ' + JSON.stringify(b.err) : '') /* 2회차: 출처별 숫자가 없어 위키미디어 전멸을 늦게 알았어요 */
}
async function curateStage() {
  const { unpack } = await import('../src/scene/descriptors.js'), all = await rd(join(WORK, 'scored.json'), [])
  const skies = all.filter((c) => c.sky && c.pass), places = dedupe(all.filter((c) => !c.sky && c.pass)).map((c) => ({ ...c, d: unpack(c.desc) }))
  const ex = A.existing ? await rd(A.existing, { scenes: [] }) : { scenes: [] }, have = new Set(ex.scenes.map((x) => x.id)), init = new Float32Array(N); for (const x of ex.scenes) { const q = unpack(x.desc); for (let i = 0; i < N; i++) init[i] += q[i] } /* 이어서 모으기: 이미 있는 장면이 채운 칸은 빼고 */
  const fresh = places.filter((c) => !have.has(c.id)), room = Math.max(0, +(A.total || 2500) - (A.existing ? 0 : 0)), { picked, gaps } = curate(fresh, { total: room, size: room + ex.scenes.length, init: ex.scenes.length ? init : null }); if (ex.scenes.length) log('이어서 모으기: 이미', ex.scenes.length, '장면 · 새 후보', fresh.length)
  log('선별:', picked.length, '/', places.length, '(중복 뺀 뒤) · 하늘', skies.length, '· 빈 칸', gaps.length); for (const g of gaps.slice(0, 12)) log('  빈 칸', g.ko, `${g.have}/${g.want}`)
  await wr(join(WORK, 'selected.json'), picked.map(({ d, ...c }) => c)); await wr(join(WORK, 'gaps.json'), { gaps, terms: gapTerms(gaps) }); await wr(join(WORK, 'skies.json'), skies)
}
async function buildStage() {
  const sel = await rd(join(WORK, 'selected.json'), []), [i, n] = String(A.shard || '0/1').split('/').map(Number), mine = sel.filter((_, k) => k % n === i), out = A.out || `out/shard-${i}`
  process.argv.push('--out', out); const F = await import('./scene-factory.mjs'), parts = []
  for (const c of mine) {
    const kind = 'photo' /* 모든 출처가 실제 장소 사진 (깊이만 AI 추정) */, top = topDesc((await import('../src/scene/descriptors.js')).unpack(c.desc), 2).map((x) => x.ko).join(' · ')
    const s = { id: c.id, kind, name: `${top} · ${c.title}`.slice(0, 60), theme: c.theme, h: 1.6, mood: null, car: false, license: c.license, author: c.author, source: c.source, title: c.title, url: c.url, desc: c.desc, stats: c.stats, blurred: c.blurred }
    try { const r = await F.build(s); if (r) parts.push({ man: r.man, rep: r.rep, pack: c.pack }) } catch (e) { log('실패 (건너뜀):', c.id, String(e.message || e).slice(0, 160)) }
  }
  await wr(join(out, 'part.json'), parts.map((p) => ({ id: p.man.id, name: p.man.name, kind: p.man.kind, label: p.man.label, theme: p.man.theme, desc: p.man.desc, stats: p.man.stats, review: p.man.review, pack: p.pack, manifest: p.man.id + '/manifest.json', sha: F.sha(JSON.stringify(p.man, null, 1)) })))
  log(`조각 ${i}/${n}:`, parts.length, '/', mine.length, '장면')
}
async function mergeStage() {
  const dir = A.partsDir || 'out', files = [], walk = async (d) => { for (const f of await readdir(d)) { const p = join(d, f); if ((await stat(p)).isDirectory()) await walk(p); else if (f === 'part.json') files.push(p) } }; await walk(dir)
  const scenes = [], hf = A['base-hf'] /* 예: 사용자/djs-scenes → https://huggingface.co/datasets/사용자/djs-scenes/resolve/main/<조각>/ */
  for (const f of files) { const shard = dirname(f).split('/').pop(); for (const s of await rd(f, [])) scenes.push({ ...s, ...(hf ? { base: `https://huggingface.co/datasets/${hf}/resolve/main/${shard}/` } : { ref: shard }) }) }
  if (A.existing) { const ex = await rd(A.existing, { scenes: [] }), have = new Set(scenes.map((x) => x.id)); for (const x of ex.scenes) if (!have.has(x.id)) scenes.push(x); log('기존 장면 합침:', ex.scenes.length) } /* 이어서 모으기: 예전 장면 그대로 */
  const skiesBase = A['skies-base'] || ''
  const index = { format: 'djs-scenes/1', ...(skiesBase ? { skiesBase } : {}), made: new Date().toISOString().slice(0, 10), count: scenes.length, scenes, skies: await rd(join(dir, 'skies', 'skies.json'), []) }
  await wr(join(A.out || 'out/scenes', 'index.json'), index); log('합치기:', scenes.length, '장면 ·', index.skies.length, '하늘 →', join(A.out || 'out/scenes', 'index.json'))
}
// 하늘 도서관: 하늘 HDRI → 1024×512 WebP + SH 9계수(재조명) + 지평선 색 + 시간대 · 날씨 서술어
async function skiesStage() {
  const sk = await rd(join(WORK, 'skies.json'), []), out = A.out || 'out/skies', list = []; await mkdir(out, { recursive: true })
  await pool(sk.slice(0, +(A.maxSkies || 60)), 4, async (c) => {
    const buf = await get(c.url), { data } = await (await img())(buf).resize(1024, 512, { fit: 'fill' }).removeAlpha().raw().toBuffer({ resolveWithObject: true }), rgb = new Uint8Array(data)
    const L = shProject((x, y) => { const i = (y * 1024 + x) * 3; return [lin(rgb[i]), lin(rgb[i + 1]), lin(rgb[i + 2])] }, 1024, 512, 4)
    let hr = 0, hg = 0, hb = 0, hn = 0; for (let y = 240; y < 272; y++) for (let x = 0; x < 1024; x += 4) { const i = (y * 1024 + x) * 3; hr += rgb[i]; hg += rgb[i + 1]; hb += rgb[i + 2]; hn++ }
    const f = `sky-${c.id}.webp`, w = await (await img())(Buffer.from(rgb), { raw: { width: 1024, height: 512, channels: 3 } }).webp({ quality: 80 }).toBuffer(); await writeFile(join(out, f), w)
    list.push({ id: c.id, file: f, title: c.title, author: c.author, license: c.L || c.license, source: c.source, sh: L.map((v) => v.map((x) => +x.toFixed(4))), horizon: [hr / hn, hg / hn, hb / hn].map((v) => Math.round(v)), desc: c.desc, sha: (await import('node:crypto')).createHash('sha256').update(w).digest('hex') })
  })
  await wr(join(out, 'skies.json'), list); log('하늘 도서관:', list.length)
}
// 합성 하늘 도서관 (앱에 묶는 시험용 · 실사 아님): 한낮 · 금빛 · 노을 · 땅거미 · 별밤 · 흐림 · 폭풍 → <out>/skies + <out>/index.json 의 skies
async function synthSkies() {
  const out = A.out || 'public/packs/scenes', W = 1024, H = 512, { dirOf } = await import('../src/scene/geom.js'), { dist } = await import('../src/scene/descriptors.js'), list = []
  const hash = (a, b) => { let h = (a * 374761393 + b * 668265263) | 0; h = (h ^ (h >>> 13)) * 1274126177; return ((h ^ (h >>> 16)) >>> 0) / 4294967296 }
  const noise = (x, y) => { const xi = Math.floor(x), yi = Math.floor(y), fx = x - xi, fy = y - yi, s = (t) => t * t * (3 - 2 * t), a = hash(xi, yi), b = hash(xi + 1, yi), c = hash(xi, yi + 1), d = hash(xi + 1, yi + 1); return (a + (b - a) * s(fx)) * (1 - s(fy)) + (c + (d - c) * s(fx)) * s(fy) }
  const fbm = (x, y) => { let v = 0, a = 0.5; for (let o = 0; o < 5; o++) { v += a * noise(x, y); x *= 2.1; y *= 2.1; a *= 0.5 } return v }
  const SK = [
    ['noon', '한낮 맑음', [0.25, 0.5, 0.95], [0.75, 0.85, 1.0], [0, 0.9, -0.4], 0.25, { 'time:noon': 2, 'weather:clear': 2, 'color:vivid': 1 }],
    ['golden', '금빛 오후', [0.35, 0.5, 0.85], [1.0, 0.8, 0.5], [0, 0.18, -1], 0.35, { 'time:golden': 2, 'weather:cloudy': 1, 'color:warm': 2 }],
    ['sunset', '노을', [0.25, 0.2, 0.5], [1.0, 0.45, 0.2], [0, 0.03, -1], 0.4, { 'time:sunset': 2.5, 'weather:cloudy': 1, 'color:warm': 2, 'color:purple': 1 }],
    ['dusk', '땅거미', [0.05, 0.08, 0.25], [0.35, 0.3, 0.55], [0, -0.05, -1], 0.2, { 'time:dusk': 2.5, 'weather:clear': 1, 'color:cool': 1.5, 'color:purple': 1 }],
    ['night', '별밤', [0.005, 0.01, 0.04], [0.04, 0.05, 0.12], [0.3, 0.5, -0.6], 0.05, { 'time:night': 2.5, 'time:midnight': 1, 'weather:clear': 1.5, 'light:moon': 1, 'color:cool': 1.5 }],
    ['overcast', '흐림', [0.55, 0.57, 0.6], [0.7, 0.71, 0.72], [0, 0.6, -0.3], 0.9, { 'weather:overcast': 2.5, 'time:noon': 0.8, 'time:morning': 0.8, 'color:mono': 2 }],
    ['storm', '폭풍 전', [0.18, 0.2, 0.25], [0.4, 0.42, 0.45], [0, 0.3, -1], 1, { 'weather:storm': 2.5, 'weather:rain': 1.5, 'time:dusk': 0.8, 'color:mono': 1.5, 'color:cool': 1 }],
  ]
  await mkdir(join(out, 'skies'), { recursive: true }); const d = [0, 0, 0]
  for (const [id, name, top, hor, sun, cloud, sc] of SK) {
    const rgb = new Uint8Array(W * H * 3), sn = Math.hypot(...sun), sd = sun.map((x) => x / sn)
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      dirOf((x + 0.5) / W, (y + 0.5) / H, d); const e = Math.max(0, d[1]), t = Math.pow(e, 0.45), i = (y * W + x) * 3
      let c = [0, 1, 2].map((k) => hor[k] + (top[k] - hor[k]) * t); const cs = Math.max(0, d[0] * sd[0] + d[1] * sd[1] + d[2] * sd[2])
      if (id !== 'night') { const g = Math.pow(cs, 600) * 30 + Math.pow(cs, 8) * 0.6; c = c.map((v, k) => v + g * [1, 0.9, 0.7][k] * (id === 'overcast' || id === 'storm' ? 0.1 : 1)) } else { const st = e > 0.05 && hash(Math.floor((Math.atan2(d[0], -d[2]) + 4) * 260), Math.floor(d[1] * 260)) > 0.996 ? 0.9 : 0, moon = Math.pow(cs, 3000) * 8; c = c.map((v) => v + st + moon) }
      if (e > 0.01) { const u = (Math.atan2(d[0], -d[2]) / Math.PI + 1) * 4, w = 1 / (e + 0.08), cl = Math.max(0, fbm(u * 2 + d[0] * w, d[2] * w * 0.6 + 10) - (1 - cloud) * 0.7) * Math.min(1, e * 6) * cloud * 1.6, cc = id === 'sunset' || id === 'golden' ? [1, 0.6, 0.45] : id === 'night' ? [0.05, 0.05, 0.08] : id === 'storm' ? [0.25, 0.26, 0.3] : [0.95, 0.95, 0.97]; c = c.map((v, k) => v * (1 - Math.min(1, cl)) + cc[k] * Math.min(1, cl)) }
      if (d[1] < 0) c = c.map((v, k) => hor[k] * 0.4)
      for (let k = 0; k < 3; k++) rgb[i + k] = Math.round(255 * Math.pow(Math.max(0, Math.min(1, c[k] / (1 + c[k]) * 1.3)), 1 / 2.2))
    }
    const L = shProject((x, y) => { const i = (y * W + x) * 3; return [lin(rgb[i]), lin(rgb[i + 1]), lin(rgb[i + 2])] }, W, H, 4)
    const f = `sky-${id}.webp`, buf = await (await img())(Buffer.from(rgb), { raw: { width: W, height: H, channels: 3 } }).webp({ quality: 80 }).toBuffer(); await writeFile(join(out, 'skies', f), buf)
    const q = dist(Object.fromEntries(Object.entries(sc).map(([k, w]) => [IDX[k], w])), 0.5), hrow = (H / 2 - 4) * W * 3; let hr = 0, hg = 0, hb = 0; for (let x = 0; x < W; x++) { hr += rgb[hrow + x * 3]; hg += rgb[hrow + x * 3 + 1]; hb += rgb[hrow + x * 3 + 2] }
    list.push({ id: 'synth-' + id, file: f, title: name + ' (합성 하늘 · 실사 아님)', author: 'AI 뮤직 DJ 스튜디오 장면 공장 (합성)', license: 'CC0', source: 'scripts/scene-mass.mjs', sh: L.map((v) => v.map((x) => +x.toFixed(4))), horizon: [hr / W, hg / W, hb / W].map(Math.round), desc: pack(q), sha: (await import('node:crypto')).createHash('sha256').update(buf).digest('hex') })
  }
  const ix = await rd(join(out, 'index.json'), { format: 'djs-scenes/1', scenes: [] }); ix.skies = list; await writeFile(join(out, 'index.json'), JSON.stringify(ix, null, 1)); log('합성 하늘:', list.length, '→', join(out, 'skies'))
}
const ST = { 'synth-skies': synthSkies, catalog, gate: gateStage, curate: curateStage, build: buildStage, merge: mergeStage, skies: skiesStage }
for (const s of String(A.stage || 'catalog').split(',')) { if (!ST[s]) throw new Error('모르는 --stage ' + s); log('==', s); await ST[s]() }
void isPano
