// v1.0-C 합성 시험 장면 (장면 공장 검증용 · 실사 아님)
//  · 작은 광선 추적기로 360° 파노라마를 그리고, 화소마다 "진짜 거리 r(m)"도 함께 내놓아요.
//    → 깊이 모델 흉내(면마다 다른 아핀 왜곡 + 잡음)를 걸어 정렬 · 지면 캘리브레이션 오차를 정답과 비교해요.
//  · 세 장면은 v1.0-C 의 세 주제(노을 도시 도로 · 겨울밤 광장 마켓 · 클럽)를 단순한 상자 · 구로 만든 것이에요.
//    실제 CC0 파노라마로 만든 팩이 오면(Actions) 그쪽을 써요. 장면 카드에는 "합성 시험 장면" 이라고 정직하게 표시해요.
import { dirOf, rng } from '../geom.js'

const hash = (a, b) => { let h = (a * 374761393 + b * 668265263) | 0; h = (h ^ (h >>> 13)) * 1274126177; return ((h ^ (h >>> 16)) >>> 0) / 4294967296 }
const mix = (a, b, k) => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k]
const sm = (a, b, x) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t) }

function hitBox(o, d, b) {
  let t0 = 1e-4, t1 = 1e9, ax = -1, sg = 0
  for (let a = 0; a < 3; a++) {
    const inv = 1 / d[a]; let ta = (b.min[a] - o[a]) * inv, tb = (b.max[a] - o[a]) * inv, s = -1
    if (ta > tb) { const q = ta; ta = tb; tb = q; s = 1 }
    if (ta > t0) { t0 = ta; ax = a; sg = s } if (tb < t1) t1 = tb; if (t0 > t1) return null
  }
  return ax < 0 ? null : { t: t0, ax, sg }
}
function hitSphere(o, d, s) {
  const ox = o[0] - s.c[0], oy = o[1] - s.c[1], oz = o[2] - s.c[2], b = ox * d[0] + oy * d[1] + oz * d[2], c = ox * ox + oy * oy + oz * oz - s.r * s.r, D = b * b - c
  if (D < 0) return null; const t = -b - Math.sqrt(D); return t > 1e-4 ? t : null
}
// 장면 하나를 그려요 → { rgb: Uint8Array(W·H·3, sRGB), r: Float32Array(W·H, 하늘 = Infinity), h }
export function renderSynth(sc, W, H) {
  const rgb = new Uint8Array(W * H * 3), R = new Float32Array(W * H), d = [0, 0, 0], o = [0, 0, 0], L = sc.sun ? norm(sc.sun) : null
  const trace = (o, d, any) => { /* 가장 가까운 물체 (any: 그림자용, 하나라도 맞으면) */
    let best = null
    if (d[1] < -1e-6) { const t = (-sc.h - o[1]) / d[1]; if (t > 1e-4) best = { t, kind: 'g' } }
    if (sc.room) { const r = sc.room; for (let a = 0; a < 3; a++) { if (Math.abs(d[a]) < 1e-9) continue; const t = ((d[a] > 0 ? r.max[a] : r.min[a]) - o[a]) / d[a]; if (t > 1e-4 && (!best || t < best.t) && !(a === 1 && d[1] < 0)) best = { t, kind: 'w', ax: a, sg: d[a] > 0 ? 1 : -1 } } }
    for (const b of sc.boxes) { const h = hitBox(o, d, b); if (h && (!best || h.t < best.t)) { best = { t: h.t, kind: 'b', b, ax: h.ax, sg: h.sg }; if (any) return best } }
    for (const s of sc.spheres) { const t = hitSphere(o, d, s); if (t && (!best || t < best.t)) { best = { t, kind: 's', s }; if (any) return best } }
    return best
  }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    dirOf((x + 0.5) / W, (y + 0.5) / H, d); const h = trace(o, d, false), i = y * W + x; let c
    if (!h) { c = sc.sky(d); R[i] = Infinity } else {
      const p = [d[0] * h.t, d[1] * h.t, d[2] * h.t]; let n, m
      if (h.kind === 'g') { n = [0, 1, 0]; m = sc.ground(p) }
      else if (h.kind === 'w') { n = [0, 0, 0]; n[h.ax] = -h.sg; m = sc.wall(p, h.ax, h.sg) }
      else if (h.kind === 'b') { n = [0, 0, 0]; n[h.ax] = h.sg; m = h.b.m(p, n) }
      else { const s = h.s; n = norm([p[0] - s.c[0], p[1] - s.c[1], p[2] - s.c[2]]); m = s.m }
      let lit = sc.amb
      if (L) { const nd = n[0] * L[0] + n[1] * L[1] + n[2] * L[2]; if (nd > 0) { const q = [p[0] + n[0] * 1e-3, p[1] + n[1] * 1e-3, p[2] + n[2] * 1e-3], sh = trace(q, L, true); if (!sh || sh.kind === 's') lit = [lit[0] + sc.sunC[0] * nd, lit[1] + sc.sunC[1] * nd, lit[2] + sc.sunC[2] * nd] } }
      const e = m.e || [0, 0, 0]; c = [m.c[0] * lit[0] + e[0], m.c[1] * lit[1] + e[1], m.c[2] * lit[2] + e[2]]
      const f = 1 - Math.exp(-h.t * sc.fogK); c = mix(c, sc.fog, f * (m.e ? 0.4 : 1)); R[i] = h.t
    }
    for (let k = 0; k < 3; k++) { const v = c[k] / (1 + c[k]); rgb[i * 3 + k] = Math.round(255 * Math.pow(Math.max(0, Math.min(1, v * 1.25)), 1 / 2.2)) }
  }
  return { rgb, r: R, h: sc.h }
}
function norm(v) { const n = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / n, v[1] / n, v[2] / n] }
const box = (x0, y0, z0, x1, y1, z1, m) => ({ min: [Math.min(x0, x1), Math.min(y0, y1), Math.min(z0, z1)], max: [Math.max(x0, x1), Math.max(y0, y1), Math.max(z0, z1)], m: typeof m === 'function' ? m : () => m })
// 창문 격자 (건물 벽면): 칸 3m × 3.4m, 켜진 창은 따뜻하게
const facade = (base, lit, seed, on = 0.45) => (p, n) => {
  if (n[1] !== 0) return { c: base }
  const u = n[0] !== 0 ? p[2] : p[0], cu = Math.floor(u / 3), cv = Math.floor(p[1] / 3.4), fu = u / 3 - cu, fv = p[1] / 3.4 - cv
  if (fu > 0.2 && fu < 0.8 && fv > 0.25 && fv < 0.8) return hash(cu + seed, cv) < on ? { c: [0.05, 0.05, 0.05], e: lit.map((x) => x * (0.6 + 0.8 * hash(cv, cu + seed))) } : { c: [0.03, 0.04, 0.06] }
  return { c: base }
}

// 1) 노을 도시 도로 — 도로(차선) · 인도 · 양쪽 건물 · 가로등 · 낮은 해
function city(seed) {
  const R = rng(seed), h = 1.6, boxes = [], spheres = []
  for (const side of [-1, 1]) {
    boxes.push(box(side * 6.2, -h, -400, side * 10, -h + 0.15, 400, { c: [0.32, 0.3, 0.3] }))
    for (let z = -300; z < 300;) { const w = 10 + R() * 16, dep = 8 + R() * 14, ht = 8 + R() * 40, x0 = side * (10 + R() * 1.5), tint = 0.12 + R() * 0.2; boxes.push(box(x0, -h, z, x0 + side * dep, -h + ht, z + w - 1.5, facade([tint, tint * 0.9, tint * 0.85], [1.6, 1.0, 0.5], (z * 7 + side * 13) | 0))); z += w }
    for (let z = -290; z < 300; z += 26) { boxes.push(box(side * 6.6 - 0.08, -h, z - 0.08, side * 6.6 + 0.08, -h + 7, z + 0.08, { c: [0.15, 0.15, 0.16] })); spheres.push({ c: [side * 5.9, -h + 7, z], r: 0.28, m: { c: [0, 0, 0], e: [6, 4.2, 2.2] } }) }
  }
  const X0 = 1.8 /* 카메라는 가운데 차선이 아니라 한 차선 안에 서요 */; for (const b of boxes) { b.min[0] -= X0; b.max[0] -= X0 } for (const s of spheres) s.c[0] -= X0
  return {
    h, boxes, spheres, sun: [0, 0.07, -1], sunC: [2.6, 1.3, 0.6], amb: [0.32, 0.24, 0.3], fog: [0.85, 0.5, 0.35], fogK: 0.004,
    ground: (p) => { const x = Math.abs(p[0] + 1.8), road = x < 6.2, dash = x < 0.12 && ((p[2] % 9) + 9) % 9 < 3, edge = Math.abs(x - 5.8) < 0.1, n = 0.04 * hash(Math.floor(p[0] * 3), Math.floor(p[2] * 3)); return { c: dash || edge ? [0.85, 0.8, 0.6] : road ? [0.09 + n, 0.09 + n, 0.1 + n] : [0.3, 0.29, 0.28] } },
    sky: (d) => { const e = Math.max(0, d[1]), sd = Math.max(0, -d[2] * 0.998 + d[1] * 0.07), c = mix([2.2, 0.9, 0.35], [0.12, 0.1, 0.32], Math.pow(e, 0.45)); const g = Math.pow(sd, 400) * 40 + Math.pow(sd, 12) * 0.8; return [c[0] + g * 1.2, c[1] + g * 0.6, c[2] + g * 0.25] },
  }
}
// 2) 겨울밤 광장 마켓 — 눈 덮인 광장 · 노점 고리 · 전구 줄 · 트리 · 둘레 건물 · 별
function market(seed) {
  const R = rng(seed), h = 1.6, boxes = [], spheres = [], N = 11
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2 + 0.2, rr = 8 + R() * 1.5, cx = Math.sin(a) * rr, cz = -Math.cos(a) * rr, col = [[0.6, 0.08, 0.06], [0.08, 0.3, 0.12], [0.5, 0.35, 0.1]][i % 3]
    boxes.push(box(cx - 1.4, -h, cz - 1.1, cx + 1.4, -h + 2.3, cz + 1.1, { c: [0.35, 0.22, 0.12], e: [0.5, 0.3, 0.1] }))
    boxes.push(box(cx - 1.6, -h + 2.3, cz - 1.3, cx + 1.6, -h + 2.7, cz + 1.3, (p) => ({ c: Math.floor((p[0] + p[2]) * 2.5) % 2 ? col : [0.85, 0.85, 0.85] })))
    const b = (i + 1) / N * Math.PI * 2 + 0.2, nx = Math.sin(b) * rr, nz = -Math.cos(b) * rr
    for (let k = 1; k < 12; k++) { const t = k / 12, sag = Math.sin(t * Math.PI) * 0.7; spheres.push({ c: [cx + (nx - cx) * t, -h + 3.6 - sag, cz + (nz - cz) * t], r: 0.07, m: { c: [0, 0, 0], e: k % 3 ? [5, 3.4, 1.5] : [1.5, 3, 5] } }) }
  }
  boxes.push(box(-0.3, -h, -16.3, 0.3, -h + 2, -15.7, { c: [0.25, 0.15, 0.08] }))
  for (let k = 0; k < 7; k++) { const y = -h + 2 + k * 1.3, w = 3.2 - k * 0.42; boxes.push(box(-w, y, -16 - w, w, y + 1.3, -16 + w, { c: [0.04, 0.22, 0.08] })); for (let j = 0; j < 8; j++) { const a = j / 8 * Math.PI * 2 + k; spheres.push({ c: [Math.sin(a) * (w + 0.05), y + 0.6, -16 - Math.cos(a) * (w + 0.05)], r: 0.09, m: { c: [0, 0, 0], e: j % 2 ? [6, 1.2, 0.8] : [6, 5, 2] } }) } }
  spheres.push({ c: [0, -h + 11.6, -16], r: 0.35, m: { c: [0, 0, 0], e: [8, 7, 3] } })
  for (let i = 0; i < 16; i++) { const a = i / 16 * Math.PI * 2, rr = 26 + R() * 4, cx = Math.sin(a) * rr, cz = -Math.cos(a) * rr, w = 5 + R() * 3, ht = 10 + R() * 12; boxes.push(box(cx - w, -h, cz - w, cx + w, -h + ht, cz + w, facade([0.18, 0.16, 0.15], [1.5, 0.95, 0.45], i * 31, 0.55))) }
  return {
    h, boxes, spheres, sun: [0.3, 0.6, 0.2], sunC: [0.12, 0.14, 0.22], amb: [0.13, 0.15, 0.24], fog: [0.06, 0.08, 0.16], fogK: 0.012,
    ground: (p) => { const n = 0.06 * hash(Math.floor(p[0] * 4), Math.floor(p[2] * 4)); return { c: [0.75 + n, 0.8 + n, 0.92 + n] } },
    sky: (d) => { const e = Math.max(0, d[1]), c = mix([0.1, 0.12, 0.25], [0.01, 0.015, 0.05], Math.pow(e, 0.5)); const sx = Math.floor((Math.atan2(d[0], -d[2]) + 4) * 300), sy = Math.floor(d[1] * 300), st = e > 0.08 && hash(sx, sy) > 0.997 ? 1.5 : 0; return [c[0] + st, c[1] + st, c[2] + st] },
  }
}
// 3) 클럽 — 실내 방 · 무대 · LED 월 · 트러스 · 천장 조명 · 스피커
function club(seed) {
  const R = rng(seed), h = 1.6, boxes = [], spheres = [], room = { min: [-12, -h, -14], max: [12, 4.4, 14] }
  boxes.push(box(-8, -h, -14, 8, -h + 1, -9.5, { c: [0.06, 0.06, 0.07] }))
  boxes.push(box(-7, -h + 1, -13.9, 7, -h + 5.2, -13.7, (p) => { const cu = Math.floor(p[0] * 3), cv = Math.floor(p[1] * 3), f = (p[0] * 3 - cu < 0.85) && (p[1] * 3 - cv < 0.85), w = 0.5 + 0.5 * Math.sin(p[0] * 0.6 + p[1] * 0.9); return f ? { c: [0, 0, 0], e: [3.5 * w + 0.4, 0.3, 3.5 * (1 - w) + 0.6] } : { c: [0.01, 0.01, 0.01] } }))
  for (const s of [-1, 1]) { boxes.push(box(s * 8.2, -h, -13.5, s * 10, -h + 3.2, -11.5, { c: [0.03, 0.03, 0.035] })); boxes.push(box(s * 8.4, -h + 0.4, -11.49, s * 9.8, -h + 2.8, -11.45, (p) => ({ c: Math.hypot(p[0] - s * 9.1, ((p[1] + h - 0.9) % 1.1) - 0.5) < 0.42 ? [0.01, 0.01, 0.01] : [0.06, 0.06, 0.07] }))) }
  for (let z = -12; z <= 10; z += 5.5) { boxes.push(box(-11, 3.6, z - 0.15, 11, 3.9, z + 0.15, { c: [0.4, 0.4, 0.42] })); for (let x = -9; x <= 9; x += 3) spheres.push({ c: [x, 3.4, z], r: 0.18, m: { c: [0, 0, 0], e: R() < 0.5 ? [6, 0.8, 4] : [0.8, 3.5, 6] } }) }
  boxes.push(box(-12, -h, 6, -6, -h + 1.1, 14, { c: [0.12, 0.07, 0.05], e: [0.25, 0.08, 0.02] }))
  return {
    h, room, boxes, spheres, sun: null, amb: [0.16, 0.1, 0.2], fog: [0.18, 0.06, 0.22], fogK: 0.035,
    ground: (p) => { const g = (Math.abs(p[0] % 2) < 0.04 || Math.abs(p[2] % 2) < 0.04) ? 0.12 : 0.04; return { c: [g, g * 0.8, g * 1.2] } },
    wall: (p, ax) => (ax === 1 ? { c: [0.05, 0.05, 0.06] } : { c: Math.floor(p[1] * 1.2) % 2 ? [0.08, 0.06, 0.1] : [0.06, 0.05, 0.08] }),
    sky: () => [0, 0, 0],
  }
}
export const SYNTH = {
  city: { make: city, name: '노을 도시 도로', mood: { energy: 0.6, bpm: [80, 120], moods: ['bright', 'calm'], genres: ['citypop', 'lofi', 'disco', 'kpop'] }, car: true },
  market: { make: market, name: '겨울밤 광장 마켓', mood: { energy: 0.45, bpm: [60, 120], moods: ['calm', 'bright'], genres: ['carol', 'waltz', 'ballad', 'lofi'] }, car: false },
  club: { make: club, name: '클럽', mood: { energy: 0.85, bpm: [118, 175], moods: ['energetic', 'dark'], genres: ['house', 'techno', 'dnb', 'trap', 'disco'] }, car: false },
}
