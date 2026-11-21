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
  vd: { url: 'https://github.com/sfzinstruments/virtuosity_drums', license: 'CC0-1.0', author: 'Versilian Studios (Virtuosity Drums · 연주 Austin McMahon)', source: 'GitHub sfzinstruments/virtuosity_drums' }, /* 2단계: 드럼 멀티샘플 (LICENSE = CC0 1.0 확인) */
}
// 등급별 설정: 세기 레이어 수 · 길이 상한(초) · 비트레이트(kbps)
const Q = {
  lite: { pianoLayers: [4, 8, 12, 16], cap: 1, kbps: { 1: 48, 2: 64 }, relEvery: 3, drum: { L: 3, R: 2, cap: 0.7 } }, /* v1.0-B.3: 40/56 → 48/64 kbps (10MB 안에서 조금 더 맑게) · 2단계 드럼: 세기 3 × 라운드로빈 2 */
  standard: { pianoLayers: [2, 5, 8, 11, 14, 16], cap: 1.6, kbps: { 1: 64, 2: 96 }, relEvery: 1, drum: { L: 5, R: 3, cap: 1 } },
  full: { pianoLayers: [1, 3, 5, 7, 9, 11, 13, 14, 15, 16], cap: 2.4, kbps: { 1: 80, 2: 104 }, relEvery: 1, drum: { L: 8, R: 4, cap: 1.3 } }, /* 등급마다 따로 가지(ref)에 올려요: jsDelivr 는 ref 하나 50MB · 파일 하나 20MB 까지 [확인 필요] */
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
/* ---------------- 2단계 드럼 멀티샘플: 킷마다 GM 음(킥 36 · 림 37 · 스네어 38 · 클랩 39 · 닫힌 햇 42 · 낮은 탐 45 · 열린 햇 46 · 크래시 49 · 높은 탐 50 · 라이드 51) × 세기 층 × 라운드로빈
   · 어쿠스틱 = Virtuosity Drums (CC0) 의 가까운 마이크 + 오버헤드 + 룸을 같은 타격끼리 섞어(마이크가 동시에 녹음돼 정렬이 맞아요) 스테레오로
   · 로파이 = 같은 녹음의 "lofi"(빈티지 마이크) 위치 + 룸 조금
   · 808 · 909 = 아날로그 회로를 흉내 낸 이 스크립트의 합성(브리지드-T 공진 · 금속성 사각파 6개 · 잡음 버스트) — 녹음이 아닌 이 프로젝트 저작물(CC0).
     세기 = 액센트(크기 · 드라이브 · 밝기), 라운드로빈 = 부품 오차(음높이 ±0.4% · 감쇠 ±4% · 잡음 씨앗)
   · 원샷: 앱 · 플러그인은 놓기 없이 끝까지 재생 · 닫힌 햇이 열린 햇을 막아요 (engine/src/lib.rs drum_sample · wdsp.js drumSample) */
const GM = { kick: 36, rim: 37, snare: 38, clap: 39, hhc: 42, ltom: 45, hho: 46, crash: 49, htom: 50, ride: 51 }
const DCAP = { kick: 1.1, rim: 0.35, snare: 0.9, clap: 0.7, hhc: 0.45, ltom: 1.1, hho: 1.4, crash: 2.4, htom: 1, ride: 2 }
const pickEven = (arr, n) => { if (arr.length <= n) return arr; const o = []; for (let i = 0; i < n; i++) o.push(arr[Math.round((i * (arr.length - 1)) / Math.max(1, n - 1))]); return [...new Set(o)] }
// Virtuosity 파일 이름: Samples/<마이크>/<부품>/<마이크>_<주법>_vl<세기>[_rr<차례>].flac
function vdZones(piece, art, mics) {
  const re = new RegExp(`^Samples/${mics[0][0]}/[a-z]+/${mics[0][0]}_${art}_vl(\\d+)(?:_rr(\\d+))?\\.flac$`), hits = list('vd').map((p) => { const m = re.exec(p); return m && { p, vl: +m[1], rr: +(m[2] || 1) } }).filter(Boolean)
  const vls = pickEven([...new Set(hits.map((h) => h.vl))].sort((a, b) => a - b), Q.drum.L), rrs = [...new Set(hits.map((h) => h.rr))].sort((a, b) => a - b).slice(0, Q.drum.R)
  const out = []; for (const h of hits) { if (!vls.includes(h.vl) || !rrs.includes(h.rr)) continue; out.push({ src: 'vd', drum: true, root: GM[piece], piece, layer: vls.indexOf(h.vl), rr: rrs.indexOf(h.rr), name: h.p.split('/').pop(), mix: mics.map(([m, w]) => ({ path: h.p.replace(new RegExp(`^Samples/${mics[0][0]}/`), `Samples/${m}/`).replace(new RegExp(`/${mics[0][0]}_`), `/${m}_`), w })) }) }
  if (!out.length) throw new Error('Virtuosity 파일 없음: ' + art)
  return out
}
function acousticKit(mixFor) {
  return () => [
    ...vdZones('kick', 'kick_snon', mixFor('kick')), ...vdZones('snare', 'snare_center', mixFor('snare')), ...vdZones('rim', 'snare_rimshot', mixFor('snare')), ...vdZones('clap', 'snare_crossstick', mixFor('snare')),
    ...vdZones('hhc', 'hh_closed', mixFor('cym')), ...vdZones('hho', 'hh_open', mixFor('cym')), ...vdZones('ltom', 'ltom_center', mixFor('tom')), ...vdZones('htom', 'htom_center', mixFor('tom')),
    ...vdZones('crash', 'crash_crash', mixFor('cym')), ...vdZones('ride', 'ride_ride', mixFor('cym')),
  ]
}
const ACOUSTIC_MIX = (k) => ({ kick: [['kickmic', 1], ['oh', 0.55], ['room', 0.35]], snare: [['snaremic', 1], ['oh', 0.55], ['room', 0.35]], tom: [['oh', 1], ['mid', 0.5], ['room', 0.35]], cym: [['oh', 1], ['room', 0.3]] })[k]
const LOFI_MIX = (k) => ({ kick: [['lofi', 1], ['room', 0.2]], snare: [['lofi', 1], ['room', 0.2]], tom: [['lofi', 1], ['room', 0.2]], cym: [['lofi', 1], ['room', 0.15]] })[k]

// 아날로그 흉내 합성 (결정적: 씨앗 = 킷 · 부품 · 층 · 차례)
function metal(n, freqs, r) { const o = new Float32Array(n), ph = freqs.map(() => r()); for (let i = 0; i < n; i++) { let s = 0; for (let k = 0; k < freqs.length; k++) { ph[k] += freqs[k] / SR; if (ph[k] >= 1) ph[k] -= 1; s += ph[k] < 0.5 ? 1 : -1 } o[i] = s / freqs.length } return o }
const M808 = [205.3, 304.4, 369.6, 522.7, 540, 800], M909 = [263, 401, 462, 597, 703, 849] /* 808 의 금속 사각파 6개 주파수(회로에서 잰 값 — 사실값) · 909 흉내는 다른 비 */
function drumSynth(model, piece, vel, seed) {
  const r = rng(seed), j = (a) => 1 + (r() * 2 - 1) * a, acc = 0.55 + 0.45 * vel, cap = DCAP[piece], n = Math.round(cap * SR * 1.05), out = new Float32Array(n), noise = () => r() * 2 - 1, is9 = model === '909'
  if (piece === 'kick') {
    const f0 = (is9 ? 52 : 47) * j(0.004), dec = (is9 ? 0.32 : 0.9) * j(0.04) * (0.8 + 0.4 * vel), sweep = is9 ? 4.2 : 1.6, st = is9 ? 0.028 : 0.012; let ph = 0
    for (let i = 0; i < n; i++) { const t = i / SR, f = f0 * (1 + sweep * Math.exp(-t / st)); ph += f / SR; out[i] = Math.sin(2 * Math.PI * ph) * Math.exp(-t / dec) }
    const ck = biquad(Float32Array.from({ length: Math.round(SR * 0.004) }, noise), 'hp', is9 ? 2500 : 1200, 0.7); for (let i = 0; i < ck.length; i++) out[i] += ck[i] * (is9 ? 0.5 : 0.25) * acc * (1 - i / ck.length)
    const drive = 1 + vel * (is9 ? 2.2 : 1.4); for (let i = 0; i < n; i++) out[i] = Math.tanh(out[i] * drive) / Math.tanh(drive)
  } else if (piece === 'snare' || piece === 'rim') {
    const rim = piece === 'rim', f1 = (rim ? 1700 : is9 ? 190 : 180) * j(0.004), f2 = (rim ? 520 : is9 ? 345 : 330) * j(0.004), td = rim ? 0.012 : is9 ? 0.09 : 0.11, nd = rim ? 0.01 : (is9 ? 0.19 : 0.16) * j(0.04)
    let p1 = 0, p2 = 0; for (let i = 0; i < n; i++) { const t = i / SR; p1 += f1 * (1 + (is9 ? 0.25 : 0.08) * Math.exp(-t / 0.01)) / SR; p2 += f2 / SR; out[i] = (Math.sin(2 * Math.PI * p1) * 0.6 + Math.sin(2 * Math.PI * p2) * 0.4) * Math.exp(-t / td) }
    if (!rim) { const nz = biquad(biquad(Float32Array.from({ length: n }, noise), 'hp', is9 ? 1200 : 1600, 0.7), 'lp', is9 ? 9000 : 7500, 0.7), snap = 0.5 + 0.5 * vel; for (let i = 0; i < n; i++) out[i] += nz[i] * Math.exp(-(i / SR) / nd) * snap * 1.1 }
  } else if (piece === 'clap') {
    const nz = biquad(Float32Array.from({ length: n }, noise), 'bp', (is9 ? 1500 : 1100) * j(0.02), 1.4), bursts = [0, 0.009, 0.019, 0.031].map((x) => x * j(0.08))
    for (let i = 0; i < n; i++) { const t = i / SR; let a = 0; for (const b of bursts) if (t >= b) a = Math.max(a, Math.exp(-(t - b) / 0.0045)); const tail = t >= bursts[3] ? Math.exp(-(t - bursts[3]) / (is9 ? 0.13 : 0.16)) * 0.55 : 0; out[i] = nz[i] * Math.max(a, tail) * 2.2 }
  } else if (piece === 'hhc' || piece === 'hho' || piece === 'crash' || piece === 'ride') {
    const dec = { hhc: 0.035 + 0.02 * vel, hho: 0.32, crash: 1.1, ride: 1.4 }[piece] * j(0.04), base = (is9 ? M909 : M808).map((f) => f * (piece === 'ride' ? 1.5 : piece === 'crash' ? 1.25 : 1) * j(0.004))
    const m = metal(n, base, r), hp = piece === 'ride' ? 5000 : piece === 'crash' ? 4000 : 7000, bp = biquad(biquad(m, 'bp', piece === 'ride' ? 8000 : 9500, 0.9), 'hp', hp, 0.7), nz = biquad(Float32Array.from({ length: n }, noise), 'hp', 8000, 0.7), nw = piece === 'crash' ? 0.6 : is9 ? 0.45 : 0.15
    for (let i = 0; i < n; i++) { const t = i / SR; out[i] = (bp[i] * (1 - nw) + nz[i] * nw) * Math.exp(-t / dec) * (piece === 'ride' && t < 0.03 ? 1.6 : 1) * 2.4 }
  } else if (piece === 'ltom' || piece === 'htom') {
    const f0 = (piece === 'ltom' ? 95 : 145) * (is9 ? 1.05 : 1) * j(0.004), dec = (is9 ? 0.28 : 0.38) * j(0.04); let ph = 0
    for (let i = 0; i < n; i++) { const t = i / SR; ph += f0 * (1 + 0.6 * Math.exp(-t / 0.02)) / SR; out[i] = Math.sin(2 * Math.PI * ph) * Math.exp(-t / dec) }
    if (is9) { const nz = biquad(Float32Array.from({ length: Math.round(SR * 0.03) }, noise), 'bp', 900, 1); for (let i = 0; i < nz.length; i++) out[i] += nz[i] * 0.3 * (1 - i / nz.length) }
  }
  for (let i = 0; i < n; i++) out[i] *= acc
  return [out] /* 모노 (회로 흉내라 좌우가 같아요 — 스테레오로 담으면 크기만 두 배) */
}
function synthKit(model) { const L = Math.min(Q.drum.L, 4), R = Math.min(Q.drum.R, 3); return () => { const z = []; let k = 0; for (const piece of Object.keys(GM)) { for (let l = 0; l < L; l++) for (let rr = 0; rr < R; rr++) z.push({ synth: true, drum: true, model, piece, root: GM[piece], layer: l, rr, vel: L === 1 ? 1 : 0.35 + 0.65 * (l / (L - 1)), seed: 7000 + (model === '909' ? 5000 : 0) + k * 131 + l * 17 + rr * 5, name: `${model}_${piece}_${l}_${rr}` }) ; k++ } return z } }
const DRUMS = [
  { id: 'd-acoustic', kit: 'acoustic', title: '어쿠스틱 드럼 (Virtuosity Drums · 가까운 + 오버헤드 + 룸 마이크)', ch: 2, drum: true, env: { a: 0.0005, r: 0.05 }, zones: acousticKit(ACOUSTIC_MIX) },
  { id: 'd-lofi', kit: 'lofi', title: '로파이 드럼 (Virtuosity Drums · 빈티지 마이크)', ch: 2, drum: true, env: { a: 0.0005, r: 0.05 }, zones: acousticKit(LOFI_MIX) },
  { id: 'd-808', kit: 'k808', title: '808 스타일 드럼 (아날로그 흉내 합성 · 자체 생성)', ch: 1, drum: true, env: { a: 0.0005, r: 0.05 }, zones: synthKit('808') },
  { id: 'd-909', kit: 'house', title: '909 스타일 드럼 (아날로그 흉내 합성 · 자체 생성)', ch: 1, drum: true, env: { a: 0.0005, r: 0.05 }, zones: synthKit('909') },
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
  const paths = {}; for (const z of zs) if (!z.synth && z.path) (paths[z.src] = paths[z.src] || []).push(z.path)
  for (const z of zs) if (z.mix) (paths[z.src] = paths[z.src] || []).push(...z.mix.map((m) => m.path))
  for (const k in paths) fetchFiles(k, paths[k])
  const raw = zs.map((z) => {
    if (z.drum && z.synth) return { z, chs: drumSynth(z.model, z.piece, z.vel, z.seed), root: z.root }
    if (z.drum) { let acc = null; for (const m of z.mix) { const c = decode(join(SRC, z.src, m.path), def.ch); if (!acc) acc = c.map((a) => a.map((v) => v * m.w)); else for (let q = 0; q < def.ch; q++) { const a = acc[q], b = c[q], L = Math.max(a.length, b.length); if (L > a.length) { const na = new Float32Array(L); na.set(a); acc[q] = na } for (let i = 0; i < b.length; i++) acc[q][i] += b[i] * m.w } } return { z, chs: acc, root: z.root } } /* 같은 타격의 마이크들을 섞어요 */
    if (z.synth) return { z, chs: guitarNote(z.midi, [0.35, 0.65, 1][z.layer], 1000 + z.midi * 31 + z.layer * 7 + z.rr * 3, def.cap(z.midi) + 0.1), root: z.midi }
    const chs = decode(join(SRC, z.src, z.path), def.ch); return { z, chs, byName: nameMidi(z.name), det: detectMidi(chs) }
  })
  const fix = def.drum ? { off: 0, votes: {} } : octaveFix(raw); for (const r of raw) if (r.root == null) r.root = r.byName != null ? r.byName + fix.off : Math.round(r.det || 60)
  if (!zs[0].synth && !def.drum) console.log(`  ${def.id}: 옥타브 보정 ${fix.off} (표: ${JSON.stringify(fix.votes)})`)
  const gpk = Math.max(...raw.map((r) => peakOf(r.chs))) || 1, norm = 0.89 / gpk
  // 스프라이트가 너무 길면 여러 파일로 나눠요 (jsDelivr 는 파일 하나 20MB 까지)
  const MAXSEC = Math.floor((12 * 1048576 * 8) / (Q.kbps[def.ch] * 1000)) /* VBR 이라 실제 크기가 15~20% 넘칠 수 있어 12MB 를 목표로 */, sprites = []; let cur = null
  for (const r of raw.sort((a, b) => a.root - b.root || a.z.layer - b.z.layer || a.z.rr - b.z.rr)) {
    const c = def.drum ? trim(r.chs.map((a) => a.map((v) => v * norm)), DCAP[r.z.piece] * Q.drum.cap, Math.min(0.08, DCAP[r.z.piece] * Q.drum.cap * 0.3)) : trim(r.chs.map((a) => a.map((v) => v * norm)), def.cap(r.root))
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
  const inst = { title: def.title, ...(def.drum ? { drum: true, kit: def.kit } : {}), file: files[0], ...(files.length > 1 ? { files } : {}), ch: def.ch, layers: Math.max(...zones.map((z) => z.layer)) + 1, rr: Math.max(...zones.map((z) => z.rr)) + 1, env: def.env, zones }
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
  for (const d of INSTR) if (!process.env.ONLY_DRUMS) instruments[d.id] = buildInstrument(d) /* ONLY_DRUMS=1: 드럼만 시험 빌드 */
  if (!process.env.NO_DRUMS) for (const d of DRUMS) instruments[d.id] = buildInstrument(d)
  for (const r of IRS) { const file = 'ir-' + r.id + '.wav'; writeFileSync(join(OUT, file), wav16(makeIR(r))); irs[r.id] = { title: r.title, file, rt60: r.rt[1], len: r.len } }
  const assets = [
    { id: 'salamander', kind: 'sample', title: 'Salamander Grand Piano V3', source: REPOS.sal.source, url: 'https://archive.org/details/SalamanderGrandPianoV3', license: REPOS.sal.license, author: REPOS.sal.author, use: 'ship' },
    { id: 'vsco2ce', kind: 'sample', title: 'VS Chamber Orchestra: Community Edition', source: REPOS.vsco.source, url: 'https://github.com/sgossner/VSCO-2-CE', license: REPOS.vsco.license, author: REPOS.vsco.author, use: 'ship' },
    { id: 'ks-guitar', kind: 'sample', title: '물리 모델 기타 (확장 카플러스-스트롱 합성)', source: '이 프로젝트 scripts/build-packs.mjs', url: 'https://github.com/sin1311/-', license: 'CC0-1.0', author: '이 프로젝트', use: 'ship' },
    { id: 'virtuosity', kind: 'sample', title: 'Virtuosity Drums', source: REPOS.vd.source, url: REPOS.vd.url, license: REPOS.vd.license, author: REPOS.vd.author, use: 'ship' },
    { id: 'gen-drums', kind: 'sample', title: '808 · 909 스타일 드럼 (아날로그 흉내 합성)', source: '이 프로젝트 scripts/build-packs.mjs', url: 'https://github.com/sin1311/-', license: 'CC0-1.0', author: '이 프로젝트', use: 'ship' },
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
  if (TIER === 'lite' && total > 12 * 1048576) throw new Error('lite 팩은 12MB 이하여야 해요') /* 2단계: 드럼 4킷이 들어가 10 → 12MB */
  void statSync
}
main()
