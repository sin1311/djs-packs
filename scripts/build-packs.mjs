// v1.0-B 음원 팩 공장: CC 멀티샘플 + 자체 생성(물리 모델 기타 · 공간 응답) → Opus 스프라이트 + 매니페스트(sha256 · 라이선스 · 거울).
//  실행: node scripts/build-packs.mjs [--tier lite|standard|full] [--out 폴더]   (ffmpeg · git 필요, 인터넷이 되는 곳에서만)
//  · lite(기본): public/packs/lite 에 써요 → 앱 묶음(아티팩트)에 실려요. 10MB 이하.
//  · standard · full: 기본 출력은 .cache/packs/<tier> → packs 가지에 올리면 jsDelivr · GitHub 거울에서 받아요(사람이 올려요).
//  원칙
//  · 저작권 곡은 쓰지 않아요. 악기 한 음씩 녹음한 CC 샘플(VSCO 2 CE = CC0, Salamander = CC BY 3.0)만.
//  · 기타 · 공간 응답(IR)은 이 스크립트가 직접 합성해요(물리 모델 · 영상원 모델) → 이 프로젝트 저작물(CC0).
//  · 모든 자산은 license.js 의 validatePack 을 통과해야 매니페스트에 들어가요.
//  · 같은 입력이면 같은 출력(결정적): 잡음은 고정 씨앗 난수, 파일 순서는 정렬.
import { execFileSync, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, writeFileSync, readFileSync, rmSync, statSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { validatePack } from '../src/learn/core/license.js'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d }
const TIER = arg('tier', 'lite'), SR = 48000
const OUT = arg('out', TIER === 'lite' ? join(ROOT, 'public/packs/lite') : join(ROOT, '.cache/packs', TIER))
const SRC = process.env.PACK_SRC || join(ROOT, '.cache/pack-src')
const REPOS = {
  vsco: { url: 'https://github.com/sgossner/VSCO-2-CE', license: 'CC0-1.0', author: 'Versilian Studios (Sam Gossner · Simon Dalzell 외)', source: 'GitHub sgossner/VSCO-2-CE' },
  sal: { url: 'https://github.com/sfzinstruments/SalamanderGrandPiano', license: 'CC-BY-3.0', author: 'Alexander Holm', source: 'GitHub sfzinstruments/SalamanderGrandPiano (원본 archive.org SalamanderGrandPianoV3)' },
}
// 등급별 설정: 세기 레이어 수 · 길이 상한(초) · 비트레이트(kbps)
const Q = {
  lite: { pianoLayers: [4, 8, 12, 16], cap: 1, kbps: { 1: 40, 2: 56 }, relEvery: 3 },
  standard: { pianoLayers: [2, 5, 8, 11, 14, 16], cap: 1.6, kbps: { 1: 64, 2: 96 }, relEvery: 1 },
  full: { pianoLayers: [1, 3, 5, 7, 9, 11, 13, 14, 15, 16], cap: 2.4, kbps: { 1: 80, 2: 104 }, relEvery: 1 }, /* 등급마다 따로 가지(ref)에 올려요: jsDelivr 는 ref 하나 50MB · 파일 하나 20MB 까지 [확인 필요] */
}[TIER]
if (!Q) throw new Error('알 수 없는 등급: ' + TIER)

/* ---------------- 원본 받기 (얕은 복제 + 필요한 파일만) */
function repo(name) {
  const dir = join(SRC, name)
  if (!existsSync(join(dir, '.git'))) { mkdirSync(SRC, { recursive: true }); execFileSync('git', ['clone', '-q', '--depth', '1', '--filter=blob:none', '--no-checkout', REPOS[name].url, dir], { stdio: 'inherit' }) }
  return dir
}
const listCache = {}
function list(name) { if (!listCache[name]) listCache[name] = execFileSync('git', ['-C', repo(name), 'ls-tree', '-r', '--name-only', 'HEAD'], { maxBuffer: 1 << 26 }).toString().trim().split('\n').sort(); return listCache[name] }
function fetchFiles(name, paths) {
  const dir = repo(name), miss = paths.filter((p) => !existsSync(join(dir, p)))
  for (let i = 0; i < miss.length; i += 80) execFileSync('git', ['-C', dir, 'checkout', 'HEAD', '--', ...miss.slice(i, i + 80)], { stdio: ['ignore', 'ignore', 'inherit'] })
  return paths.map((p) => join(dir, p))
}

/* ---------------- 오디오 입출력 (ffmpeg) */
function decode(file, ch) {
  const r = spawnSync('ffmpeg', ['-v', 'error', '-i', file, '-ac', String(ch), '-ar', String(SR), '-f', 'f32le', '-'], { maxBuffer: 1 << 29 })
  if (r.status !== 0) throw new Error('ffmpeg 해독 실패: ' + file + ' ' + r.stderr)
  const b = r.stdout, f = new Float32Array(b.buffer, b.byteOffset, b.length / 4), out = []
  const n = f.length / ch; for (let c = 0; c < ch; c++) { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = f[i * ch + c]; out.push(a) }
  return out
}
function interleave(chs) { const n = chs[0].length, ch = chs.length, o = new Float32Array(n * ch); for (let c = 0; c < ch; c++) for (let i = 0; i < n; i++) o[i * ch + c] = chs[c][i]; return o }
function encodeOpus(chs, file, kbps) {
  const pcm = Buffer.from(interleave(chs).buffer)
  const r = spawnSync('ffmpeg', ['-v', 'error', '-y', '-f', 'f32le', '-ar', String(SR), '-ac', String(chs.length), '-i', '-', '-c:a', 'libopus', '-b:a', kbps + 'k', '-vbr', 'on', '-compression_level', '10', '-application', 'audio', '-fflags', '+bitexact', '-flags:a', '+bitexact', file] /* 같은 입력 → 같은 바이트 (Ogg 일련번호를 무작위로 두지 않아요) */, { input: pcm, maxBuffer: 1 << 26 })
  if (r.status !== 0) throw new Error('Opus 인코딩 실패: ' + r.stderr)
}
function wav16(chs) {
  const n = chs[0].length, ch = chs.length, b = Buffer.alloc(44 + n * ch * 2)
  b.write('RIFF', 0); b.writeUInt32LE(36 + n * ch * 2, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(ch, 22); b.writeUInt32LE(SR, 24); b.writeUInt32LE(SR * ch * 2, 28); b.writeUInt16LE(ch * 2, 32); b.writeUInt16LE(16, 34); b.write('data', 36); b.writeUInt32LE(n * ch * 2, 40)
  for (let i = 0; i < n; i++) for (let c = 0; c < ch; c++) { const x = Math.max(-1, Math.min(1, chs[c][i])); b.writeInt16LE(Math.round(x * 32767), 44 + (i * ch + c) * 2) }
  return b
}

/* ---------------- 다듬기: 앞 무음 자르기 · 길이 상한 + 끝 페이드 · 음높이 */
const peakOf = (chs) => { let p = 0; for (const a of chs) for (let i = 0; i < a.length; i++) { const v = Math.abs(a[i]); if (v > p) p = v } return p }
function trim(chs, capSec, fadeSec = 0.08) {
  const p = peakOf(chs) || 1, th = p * 0.004, n = chs[0].length
  let s = 0; while (s < n && !chs.some((a) => Math.abs(a[s]) > th)) s++
  s = Math.max(0, s - Math.round(SR * 0.002))
  let e = n; const quiet = p * 0.0008; while (e > s + SR * 0.05 && !chs.some((a) => Math.abs(a[e - 1]) > quiet)) e--
  e = Math.min(e, s + Math.round(capSec * SR)); const f = Math.min(Math.round(fadeSec * SR), e - s)
  return chs.map((a) => { const o = a.slice(s, e); for (let i = 0; i < f; i++) o[o.length - 1 - i] *= (i / f) ** 1.5; return o })
}
// YIN (de Cheveigné & Kawahara 2002) — 공격음이 지난 뒤 한 창에서 기본 주파수
function yin(x, fmin = 25, fmax = 2200) {
  const W = 4096, st = Math.min(Math.max(0, x.length - W * 2), Math.round(SR * 0.12)); if (x.length < st + W * 2) return 0
  const tmax = Math.min(W, Math.floor(SR / fmin)), tmin = Math.floor(SR / fmax), d = new Float64Array(tmax)
  for (let t = 1; t < tmax; t++) { let s = 0; for (let i = 0; i < W; i++) { const v = x[st + i] - x[st + i + t]; s += v * v } d[t] = s }
  let run = 0; const cm = new Float64Array(tmax); cm[0] = 1; for (let t = 1; t < tmax; t++) { run += d[t]; cm[t] = run ? (d[t] * t) / run : 1 }
  let tau = -1; for (let t = tmin; t < tmax - 1; t++) if (cm[t] < 0.15) { while (t + 1 < tmax && cm[t + 1] < cm[t]) t++; tau = t; break }
  if (tau < 0) { let b = tmin; for (let t = tmin; t < tmax; t++) if (cm[t] < cm[b]) b = t; if (cm[b] > 0.4) return 0; tau = b }
  const a = cm[tau - 1], b = cm[tau], c = cm[tau + 1] ?? b, den = a - 2 * b + c, fr = den ? (a - c) / (2 * den) : 0
  return SR / (tau + fr)
}
const NOTE = { C: 0, 'C#': 1, Db: 1, D: 2, 'D#': 3, Eb: 3, E: 4, F: 5, 'F#': 6, Gb: 6, G: 7, 'G#': 8, Ab: 8, A: 9, 'A#': 10, Bb: 10, B: 11 }
const nameMidi = (nm) => { const m = /([A-G](?:#|b)?)(-?\d)/.exec(nm); return m ? 12 * (+m[2] + 1) + NOTE[m[1]] : null }
// 이름의 음이름(피치 클래스)은 믿고, 옥타브는 실제 음높이로 고쳐요 (VSCO 는 악기마다 옥타브 표기가 달라요).
//  한 음씩 재면 YIN 이 가끔 한 옥타브 틀리니, 악기 전체에서 가장 흔한 옥타브 차이(최빈값) 하나를 모든 존에 써요.
function detectMidi(chs) { const f = yin(chs[0]); return f ? 69 + 12 * Math.log2(f / 440) : null }
function octaveFix(items) {
  const votes = {}; for (const it of items) { if (it.byName == null || it.det == null) continue; const o = Math.round((it.det - it.byName) / 12) * 12; votes[o] = (votes[o] || 0) + 1 }
  const best = +(Object.entries(votes).sort((a, b) => b[1] - a[1])[0] || [0])[0]
  return { off: best, votes }
}

/* ---------------- 결정적 난수 · 필터 (자체 생성 소리용) */
function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296 } }
function biquad(x, type, f, q = 0.707, gainDb = 0) {
  const w = 2 * Math.PI * f / SR, cs = Math.cos(w), sn = Math.sin(w), al = sn / (2 * q), A = 10 ** (gainDb / 40); let b0, b1, b2, a0, a1, a2
  if (type === 'lp') { b0 = (1 - cs) / 2; b1 = 1 - cs; b2 = b0; a0 = 1 + al; a1 = -2 * cs; a2 = 1 - al }
  else if (type === 'hp') { b0 = (1 + cs) / 2; b1 = -(1 + cs); b2 = b0; a0 = 1 + al; a1 = -2 * cs; a2 = 1 - al }
  else if (type === 'bp') { b0 = al; b1 = 0; b2 = -al; a0 = 1 + al; a1 = -2 * cs; a2 = 1 - al }
  else { b0 = 1 + al * A; b1 = -2 * cs; b2 = 1 - al * A; a0 = 1 + al / A; a1 = -2 * cs; a2 = 1 - al / A }
  const o = new Float32Array(x.length); let x1 = 0, x2 = 0, y1 = 0, y2 = 0
  for (let i = 0; i < x.length; i++) { const y = (b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0; x2 = x1; x1 = x[i]; y2 = y1; y1 = y; o[i] = y }
  return o
}

/* ---------------- 물리 모델 기타 (확장 카플러스-스트롱: Jaffe & Smith 1983) */
//  · 들뜸 = 세기에 따라 밝기가 다른 잡음 펄스 → 피크 위치 빗살 필터(줄의 1/7 지점을 튕긴 효과)
//  · 줄 = 지연선 + 2탭 감쇠 필터(높은 배음이 먼저 줄어요) + 1차 전역 통과 필터(소수 지연 = 정확한 음정)
//  · 몸통 = 공명 3개(약 105 · 210 · 420 Hz)를 섞어 나무 울림을 흉내
function guitarNote(midi, vel, seed, secs) {
  const f0 = 440 * 2 ** ((midi - 69) / 12), n = Math.round(secs * SR), r = rng(seed), P = SR / f0
  const S = 0.5, loopDelay = P - 0.5, N = Math.floor(loopDelay - 0.1), frac = loopDelay - N, C = (1 - frac) / (1 + frac)
  const decaySec = 2.6 * Math.pow(110 / f0, 0.45), rho = Math.pow(0.001, 1 / (decaySec * f0))
  let ex = new Float32Array(N); for (let i = 0; i < N; i++) ex[i] = r() * 2 - 1
  ex = biquad(ex, 'lp', 900 + 5200 * vel * vel, 0.6); const pick = Math.max(1, Math.round(N / 7)); const ex2 = new Float32Array(N); for (let i = 0; i < N; i++) ex2[i] = ex[i] - (i >= pick ? ex[i - pick] : 0)
  const buf = new Float32Array(N), out = new Float32Array(n); for (let i = 0; i < N; i++) buf[i] = ex2[i]
  let k = 0, prev = 0, apX = 0, apY = 0
  for (let i = 0; i < n; i++) { const x = buf[k], lp = rho * ((1 - S) * x + S * prev); prev = x; const y = C * lp + apX - C * apY; apX = lp; apY = y; buf[k] = y; out[i] = x; k = (k + 1) % N }
  const body = [[105, 6, 0.5], [210, 8, 0.35], [420, 10, 0.2]].reduce((acc, [f, q, g]) => { const b = biquad(out, 'bp', f, q); for (let i = 0; i < n; i++) acc[i] += b[i] * g * 4; return acc }, new Float32Array(n))
  const o = new Float32Array(n), atk = Math.round(SR * 0.0015); for (let i = 0; i < n; i++) o[i] = (out[i] * 0.8 + body[i] * 0.35) * (i < atk ? i / atk : 1)
  return [biquad(o, 'hp', 70, 0.7)]
}

/* ---------------- 공간 응답 IR (영상원 초기 반사 + 대역별 지수 감쇠 잔향, 스테레오 무상관) */
//  방 크기(m) · 대역별 RT60(초) 로 정해요. 녹음한 IR 이 아니라 물리 모델로 만든 이 프로젝트의 IR 이에요(CC0).
function makeIR({ dims, rt, len, pre = 0.01, seed }) {
  const n = Math.round(len * SR), out = [new Float32Array(n), new Float32Array(n)], [Lx, Ly, Lz] = dims, src = [Lx * 0.3, Ly * 0.4, 1.4], c = 343
  for (let ch = 0; ch < 2; ch++) {
    const mic = [Lx * 0.62 + (ch ? 0.09 : -0.09), Ly * 0.58, 1.5], R = rng(seed + ch * 7919)
    for (let a = -3; a <= 3; a++) for (let b = -3; b <= 3; b++) for (let e = -2; e <= 2; e++) {
      const ord = Math.abs(a) + Math.abs(b) + Math.abs(e); if (ord > 3) continue
      const ix = a * Lx + (a % 2 ? Lx - src[0] : src[0]), iy = b * Ly + (b % 2 ? Ly - src[1] : src[1]), iz = e * Lz + (e % 2 ? Lz - src[2] : src[2])
      const d = Math.hypot(ix - mic[0], iy - mic[1], iz - mic[2]), t = Math.round((pre + d / c) * SR); if (t >= n) continue
      out[ch][t] += (Math.pow(0.78, ord) / Math.max(1, d)) * (ord ? (R() < 0.5 ? -1 : 1) : 1)
    }
    const t0 = Math.round((pre + 0.012) * SR), bands = [['lp', 400, rt[0]], ['bp', 1400, rt[1]], ['hp', 4000, rt[2]]]
    for (const [ty, f, T] of bands) {
      const nz = new Float32Array(n); for (let i = t0; i < n; i++) nz[i] = R() * 2 - 1
      const fl = biquad(nz, ty, f, ty === 'bp' ? 0.6 : 0.707), k = Math.log(1000) / T
      for (let i = t0; i < n; i++) { const tt = (i - t0) / SR, ramp = Math.min(1, tt / 0.04); out[ch][i] += fl[i] * Math.exp(-k * tt) * ramp * 0.09 }
    }
    const f = Math.round(0.05 * SR); for (let i = 0; i < f; i++) out[ch][n - 1 - i] *= i / f
  }
  let e = 0; for (const a of out) for (let i = 0; i < n; i++) e += a[i] * a[i]
  const g = 1 / Math.sqrt(e / 2); for (const a of out) for (let i = 0; i < n; i++) a[i] *= g * 0.5
  return out
}

/* ---------------- 악기 정의 */
//  zones: [{ file, name, layer, rr }] → 다듬고 음높이를 재서 스프라이트에 넣어요
const capFor = (base) => (midi) => base * Q.cap * (midi < 48 ? 1.25 : midi > 84 ? 0.6 : 1)
function vscoZones(re, layerOf, rrOf) { return list('vsco').filter((p) => re.test(p)).map((p) => ({ src: 'vsco', path: p, name: p.split('/').pop(), layer: layerOf(p), rr: rrOf ? rrOf(p) : 0 })) }
const vL = (p) => +((/_v(\d)/.exec(p) || [0, 1])[1])
const rrN = (p) => +((/rr(\d)/.exec(p) || [0, 1])[1]) - 1
function denseLayers(zs) { const ls = [...new Set(zs.map((z) => z.layer))].sort((a, b) => a - b); return zs.map((z) => ({ ...z, layer: ls.indexOf(z.layer) })) }
const INSTR = [
  { id: 'piano', title: '그랜드 피아노 (Salamander)', ch: 2, cap: capFor(3.2), env: { a: 0.001, r: 0.35 }, mono: false,
    zones: () => denseLayers(list('sal').filter((p) => /^Samples\/[A-G]#?\d+v\d+\.flac$/.test(p)).filter((p) => Q.pianoLayers.includes(+/v(\d+)\./.exec(p)[1])).map((p) => ({ src: 'sal', path: p, name: p.split('/').pop(), layer: +/v(\d+)\./.exec(p)[1], rr: 0 }))),
    release: () => list('sal').filter((p) => /^Samples\/rel\d+\.flac$/.test(p)).filter((p) => (+/rel(\d+)/.exec(p)[1] - 1) % Q.relEvery === 0).map((p) => ({ src: 'sal', path: p, name: p, midi: 20 + +/rel(\d+)/.exec(p)[1] })) },
  { id: 'strings', title: '현악 앙상블 (VSCO 바이올린 · 첼로 섹션)', ch: 2, cap: capFor(4.2), env: { a: 0.06, r: 0.45 },
    zones: () => denseLayers([...vscoZones(/Strings\/Violin Section\/susVib\//, vL), ...vscoZones(/Strings\/Cello Section\/susvib\/susvib_[A-G]#?\d_v[13]_1/, (p) => (vL(p) === 3 ? 2 : 1))]).map((z) => ({ ...z, group: /Cello/.test(z.path) ? 'lo' : 'hi' })) },
  { id: 'flute', title: '플루트 (VSCO)', ch: 1, cap: capFor(3), env: { a: 0.03, r: 0.25 }, zones: () => vscoZones(/Woodwinds\/Flute\/susvib\//, () => 0, (p) => +((/_(\d)\.wav/.exec(p) || [0, 1])[1]) - 1) },
  { id: 'brass', title: '트럼펫 (VSCO)', ch: 1, cap: capFor(2.6), env: { a: 0.02, r: 0.22 }, zones: () => denseLayers(vscoZones(/Brass\/Trumpet\/sus\//, vL)) },
  { id: 'harp', title: '하프 (VSCO)', ch: 1, cap: capFor(3), env: { a: 0.001, r: 0.6 }, zones: () => vscoZones(/Strings\/Harp\/KSHarp_/, () => 0) },
  { id: 'upright', title: '콘트라베이스 피치카토 (VSCO)', ch: 1, cap: capFor(1.8), env: { a: 0.002, r: 0.18 }, zones: () => denseLayers(vscoZones(/Strings\/Solo Contrabass\/Pizz\//, vL, rrN)) },
  { id: 'guitar', title: '어쿠스틱 기타 (물리 모델 · 자체 생성)', ch: 1, cap: () => 2.4 * Math.min(1.3, Q.cap), env: { a: 0.001, r: 0.2 }, synth: true,
    zones: () => { const z = []; for (let m = 40; m <= 79; m += 3) for (let l = 0; l < 3; l++) for (let rr = 0; rr < 2; rr++) z.push({ synth: true, midi: m, layer: l, rr, name: `ks_${m}_${l}_${rr}` }); return z } },
]
const IRS = [
  { id: 'room', title: '스튜디오 룸', dims: [6, 4.5, 3], rt: [0.7, 0.55, 0.35], len: 0.9, seed: 11 },
  { id: 'hall', title: '콘서트 홀', dims: [32, 22, 14], rt: [2.6, 2.1, 1.2], len: 2.8, pre: 0.022, seed: 23 },
  { id: 'plate', title: '플레이트', dims: [3, 2, 1.2], rt: [1.6, 1.5, 1.1], len: 1.8, pre: 0.004, seed: 37 },
]

/* ---------------- 조립 */
const sha = (b) => createHash('sha256').update(b).digest('hex')
function buildInstrument(def) {
  const zs = def.zones(), gap = Math.round(SR * 0.08), zones = []
  if (!zs.length) throw new Error(def.id + ': 원본 샘플이 없어요')
  const paths = {}; for (const z of zs) if (!z.synth) (paths[z.src] = paths[z.src] || []).push(z.path)
  for (const k in paths) fetchFiles(k, paths[k])
  const raw = zs.map((z) => {
    if (z.synth) return { z, chs: guitarNote(z.midi, [0.35, 0.65, 1][z.layer], 1000 + z.midi * 31 + z.layer * 7 + z.rr * 3, def.cap(z.midi) + 0.1), root: z.midi }
    const chs = decode(join(SRC, z.src, z.path), def.ch); return { z, chs, byName: nameMidi(z.name), det: detectMidi(chs) }
  })
  const fix = octaveFix(raw); for (const r of raw) if (r.root == null) r.root = r.byName != null ? r.byName + fix.off : Math.round(r.det || 60)
  if (!zs[0].synth) console.log(`  ${def.id}: 옥타브 보정 ${fix.off} (표: ${JSON.stringify(fix.votes)})`)
  const gpk = Math.max(...raw.map((r) => peakOf(r.chs))) || 1, norm = 0.89 / gpk
  // 스프라이트가 너무 길면 여러 파일로 나눠요 (jsDelivr 는 파일 하나 20MB 까지)
  const MAXSEC = Math.floor((12 * 1048576 * 8) / (Q.kbps[def.ch] * 1000)) /* VBR 이라 실제 크기가 15~20% 넘칠 수 있어 12MB 를 목표로 */, sprites = []; let cur = null
  for (const r of raw.sort((a, b) => a.root - b.root || a.z.layer - b.z.layer || a.z.rr - b.z.rr)) {
    const c = trim(r.chs.map((a) => a.map((v) => v * norm)), def.cap(r.root))
    if (!cur || (cur.t + c[0].length) / SR > MAXSEC) { cur = { parts: [], t: gap }; sprites.push(cur) }
    zones.push({ root: r.root, layer: r.z.layer, rr: r.z.rr || 0, t0: +(cur.t / SR).toFixed(5), len: +(c[0].length / SR).toFixed(5), ...(sprites.length > 1 ? { f: sprites.length - 1 } : {}), ...(r.z.group ? { group: r.z.group } : {}) })
    cur.parts.push(c); cur.t += c[0].length + gap
  }
  const files = sprites.map((sp, k) => {
    const chs = Array.from({ length: def.ch }, () => new Float32Array(sp.t)); let o = gap
    for (const c of sp.parts) { for (let q = 0; q < def.ch; q++) chs[q].set(c[q], o); o += c[0].length + gap }
    const file = def.id + (sprites.length > 1 ? '-' + k : '') + '.ogg'; encodeOpus(chs, join(OUT, file), Q.kbps[def.ch]); return file
  })
  const total = sprites.reduce((a, sp) => a + sp.t, 0)
  const inst = { title: def.title, file: files[0], ...(files.length > 1 ? { files } : {}), ch: def.ch, layers: Math.max(...zones.map((z) => z.layer)) + 1, rr: Math.max(...zones.map((z) => z.rr)) + 1, env: def.env, zones }
  if (def.release) {
    const rs = def.release(); fetchFiles('sal', rs.map((r) => r.path)); const rp = [], rz = []; let rt = gap
    for (const r of rs) { const c = trim(decode(join(SRC, 'sal', r.path), def.ch).map((a) => a.map((v) => v * norm * 2.5)), 0.6, 0.15); rp.push(c); rz.push({ root: r.midi, t0: +(rt / SR).toFixed(5), len: +(c[0].length / SR).toFixed(5) }); rt += c[0].length + gap }
    const rc = Array.from({ length: def.ch }, () => new Float32Array(rt)); let ro = gap; for (const c of rp) { for (let k = 0; k < def.ch; k++) rc[k].set(c[k], ro); ro += c[0].length + gap }
    const rf = def.id + '-rel.ogg'; encodeOpus(rc, join(OUT, rf), Q.kbps[def.ch]); inst.release = { file: rf, zones: rz, gain: 0.35 }
  }
  console.log(`  ${def.id}: 존 ${zones.length} · 레이어 ${inst.layers} · 라운드 로빈 ${inst.rr} · ${(total / SR).toFixed(1)}초`)
  return inst
}

function main() {
  if (existsSync(OUT)) rmSync(OUT, { recursive: true }); mkdirSync(OUT, { recursive: true })
  console.log(`음원 팩 만들기: ${TIER} → ${OUT}`)
  const instruments = {}, irs = {}
  for (const d of INSTR) instruments[d.id] = buildInstrument(d)
  for (const r of IRS) { const file = 'ir-' + r.id + '.wav'; writeFileSync(join(OUT, file), wav16(makeIR(r))); irs[r.id] = { title: r.title, file, rt60: r.rt[1], len: r.len } }
  const assets = [
    { id: 'salamander', kind: 'sample', title: 'Salamander Grand Piano V3', source: REPOS.sal.source, url: 'https://archive.org/details/SalamanderGrandPianoV3', license: REPOS.sal.license, author: REPOS.sal.author, use: 'ship' },
    { id: 'vsco2ce', kind: 'sample', title: 'VS Chamber Orchestra: Community Edition', source: REPOS.vsco.source, url: 'https://github.com/sgossner/VSCO-2-CE', license: REPOS.vsco.license, author: REPOS.vsco.author, use: 'ship' },
    { id: 'ks-guitar', kind: 'sample', title: '물리 모델 기타 (확장 카플러스-스트롱 합성)', source: '이 프로젝트 scripts/build-packs.mjs', url: 'https://github.com/sin1311/-', license: 'CC0-1.0', author: '이 프로젝트', use: 'ship' },
    { id: 'gen-ir', kind: 'ir', title: '공간 응답 3종 (영상원 + 지수 감쇠 합성)', source: '이 프로젝트 scripts/build-packs.mjs', url: 'https://github.com/sin1311/-', license: 'CC0-1.0', author: '이 프로젝트', use: 'ship' },
  ]
  const files = {}
  for (const f of [...Object.values(instruments).flatMap((i) => [...(i.files || [i.file]), i.release && i.release.file]).filter(Boolean), ...Object.values(irs).map((i) => i.file)]) { const b = readFileSync(join(OUT, f)); files[f] = { sha256: sha(b), bytes: b.length } }
  const man = { format: 'djs-pack/1', id: 'core-' + TIER, tier: TIER, pack: 'core', license: 'CC-BY-3.0', sr: SR, codec: 'opus', built: 'scripts/build-packs.mjs', assets, files, instruments, irs }
  const v = validatePack(man); if (!v.ok) throw new Error('라이선스 검사 실패: ' + v.problems.join(' / '))
  man.credits = v.credits
  writeFileSync(join(OUT, 'manifest.json'), JSON.stringify(man))
  const total = Object.values(files).reduce((a, f) => a + f.bytes, 0)
  console.log(`끝: 파일 ${Object.keys(files).length}개 · ${(total / 1048576).toFixed(2)} MB`)
  if (TIER === 'lite' && total > 10 * 1048576) throw new Error('lite 팩은 10MB 이하여야 해요')
  void statSync
}
main()
