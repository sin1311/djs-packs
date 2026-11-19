// 파노라마 수평 맞추기 · 가짜 360° 거르기 (선분 검출 없이, 경계 화소마다 구면 위 "큰 원의 법선"으로)
//  · 경계 화소 p 의 방향 d, 그 자리의 경계 접선 t(밝기 기울기에 수직) → 그 직선이 놓인 큰 원의 법선 n = d × t.
//  · 실제 세계의 세로선은 모두 같은 위쪽 소실점 v 를 지나요 → n·v = 0. v = argmin Σ wᵢ (nᵢ·v)² = 행렬 Σ wᵢ nᵢnᵢᵀ 의 가장 작은
//    고윳값 고유벡터 (반복 재가중 최소제곱: 처음엔 넓게, 점점 좁게).
//  · 카메라가 기울면 수평선이 등장방형에서 y = atan(tan β · sin(x − x₀)) 사인 물결이 돼요 → v 를 위(0,1,0)로 돌리는 회전으로 바로 펴요.
//  · 가짜 360°(일부만 찍은 파노라마 · 원통형을 2:1 로 늘린 것): 진짜 등장방형에서는 세계의 직선이 구 위 큰 원에 놓여요.
//    수평을 맞춘 뒤, 가로 직선들이 수평면의 두 직교 소실점 h₁ ⟂ h₂ 를 지나는 큰 원에 얼마나 맞는지(맨해튼 점수)로 재요.
//    세로선이 충분할 때(건물 · 도시)만 판정해요 (숲 · 들판은 직선이 적어서 판정하지 않아요).
import { dirOf } from '../geom.js'

const lum = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b
// 경계 화소 → { n: Float32Array(3k), w: Float32Array(k) }. rgb 는 W×H 등장방형 (보통 1024×512)
export function edgeNormals(rgb, W, H, { stride = 2, latMax = 55, minMag = 48 } = {}) {
  const g = new Float32Array(W * H); for (let i = 0; i < W * H; i++) g[i] = lum(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2])
  const N = [], Wt = [], AZ = [], d = [0, 0, 0], a = [0, 0, 0], b = [0, 0, 0], c = [0, 0, 0], e = [0, 0, 0]
  const y0 = Math.ceil(H * (0.5 - latMax / 180)), y1 = Math.floor(H * (0.5 + latMax / 180))
  for (let y = Math.max(1, y0); y < Math.min(H - 1, y1); y += stride) for (let x = 0; x < W; x += stride) {
    const xl = (x - 1 + W) % W, xr = (x + 1) % W, G = (xx, yy) => g[yy * W + xx]
    const gx = (G(xr, y - 1) + 2 * G(xr, y) + G(xr, y + 1)) - (G(xl, y - 1) + 2 * G(xl, y) + G(xl, y + 1)), gy = (G(xl, y + 1) + 2 * G(x, y + 1) + G(xr, y + 1)) - (G(xl, y - 1) + 2 * G(x, y - 1) + G(xr, y - 1))
    const m = Math.hypot(gx, gy) / 4; if (m < minMag) continue
    const u = (x + 0.5) / W, v = (y + 0.5) / H; dirOf(u, v, d); dirOf(u + 0.5 / W, v, a); dirOf(u - 0.5 / W, v, b); dirOf(u, v + 0.5 / H, c); dirOf(u, v - 0.5 / H, e)
    const ex = [a[0] - b[0], a[1] - b[1], a[2] - b[2]], ey = [c[0] - e[0], c[1] - e[1], c[2] - e[2]], lx = Math.hypot(...ex), ly = Math.hypot(...ey)
    const tx = -gy / m, ty = gx / m /* 경계 접선 = 기울기에 수직 (화면 좌표) */, t = [0, 1, 2].map((k) => tx * ex[k] / lx + ty * ey[k] / ly)
    const n = [d[1] * t[2] - d[2] * t[1], d[2] * t[0] - d[0] * t[2], d[0] * t[1] - d[1] * t[0]], nl = Math.hypot(...n); if (nl < 1e-6) continue
    N.push(n[0] / nl, n[1] / nl, n[2] / nl); Wt.push(m); AZ.push(u)
  }
  return { n: Float32Array.from(N), w: Float32Array.from(Wt), az: Float32Array.from(AZ) }
}
// 3×3 대칭 행렬의 고유분해 (야코비) → 가장 작은 고윳값의 고유벡터
function smallestEig(S) {
  const A = S.map((r) => r.slice()), V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  for (let it = 0; it < 40; it++) {
    let p = 0, q = 1; for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) if (Math.abs(A[i][j]) > Math.abs(A[p][q])) { p = i; q = j }
    if (Math.abs(A[p][q]) < 1e-12) break
    const th = 0.5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]), c = Math.cos(th), s = Math.sin(th)
    for (let k = 0; k < 3; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq }
    for (let k = 0; k < 3; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk }
    for (let k = 0; k < 3; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq }
  }
  let m = 0; for (let i = 1; i < 3; i++) if (A[i][i] < A[m][m]) m = i
  return [V[0][m], V[1][m], V[2][m]]
}
// 위쪽 소실점 v (세계의 "위"가 사진에서 어느 방향인지) → { v, tilt(도), conf(세로선 무게 비율), ok }
export function upVector(E, { maxTilt = 20 } = {}) {
  const { n, w } = E, k = w.length; let v = [0, 1, 0], tot = 0; for (let i = 0; i < k; i++) tot += w[i]
  let inl = 0
  for (const thr of [0.2, 0.12, 0.07, 0.045, 0.03, 0.03]) {
    const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; inl = 0
    for (let i = 0; i < k; i++) { const a = n[i * 3], b = n[i * 3 + 1], c = n[i * 3 + 2]; if (Math.abs(a * v[0] + b * v[1] + c * v[2]) > thr) continue; const ww = w[i]; inl += ww; S[0][0] += ww * a * a; S[0][1] += ww * a * b; S[0][2] += ww * a * c; S[1][1] += ww * b * b; S[1][2] += ww * b * c; S[2][2] += ww * c * c }
    if (inl <= 0) break; S[1][0] = S[0][1]; S[2][0] = S[0][2]; S[2][1] = S[1][2]
    const e = smallestEig(S); if (e[1] < 0) { e[0] = -e[0]; e[1] = -e[1]; e[2] = -e[2] } const tilt = Math.acos(Math.min(1, e[1])) * 180 / Math.PI
    if (tilt > maxTilt) break /* 너무 크게 기운 답은 믿지 않아요 (세로선이 아닌 무늬에 끌려간 것) */
    v = e
  }
  const tilt = Math.acos(Math.min(1, v[1])) * 180 / Math.PI, conf = tot > 0 ? inl / tot : 0
  return { v, tilt, conf, ok: conf >= 0.12 }
}
/* 2판 (4회차 실제 자료: 세로선 비율 중앙값 0.045 — 나뭇잎 · 바닥 무늬 같은 잡동사니 경계가 많아 반복 재가중이 처음 추정(곧장 위)에 끌려
   10° 기운 사진을 1.4~6.7° 로 덜 쟀어요). 전역 탐색으로 바꿔요:
   · RANSAC: 세로선 후보 둘의 법선 nᵢ × nⱼ 가 소실점 후보 v. 좁은 허용(|n·v| < 0.012)으로 무게 합(합의)을 재서 가장 큰 것 → 최소제곱으로 다듬기.
     잡동사니는 방향이 고르게 퍼져 어느 v 에도 ε 비율만 맞지만, 진짜 세로선은 한 점에 몰려요.
   · 믿음: 왼쪽 반구 · 오른쪽 반구 경계로 따로 찾은 v 가 1°(또는 기울기의 15%) 안에서 같고, 각 반구에 맞는 경계가 충분해야 바로잡아요. */
function rnd(seed) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296 } }
function ransacUp(E, sel, { iters = 700, tol = 0.012, maxTilt = 25, seed = 7 } = {}) {
  const { n, w } = E, cand = sel.filter((i) => Math.abs(n[i * 3 + 1]) < Math.sin((maxTilt * Math.PI) / 180) + 0.02)
  if (cand.length < 20) return null
  const cw = []; let acc = 0; for (const i of cand) { acc += w[i]; cw.push(acc) }
  const R = rnd(seed), draw = () => { const r = R() * acc; let lo = 0, hi = cw.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (cw[m] < r) lo = m + 1; else hi = m } return cand[lo] }
  const cons = (v, t) => { let s = 0, c = 0; for (const i of sel) if (Math.abs(n[i * 3] * v[0] + n[i * 3 + 1] * v[1] + n[i * 3 + 2] * v[2]) < t) { s += w[i]; c++ } return [s, c] }
  let best = null, bs = -1, cosMax = Math.cos((maxTilt * Math.PI) / 180)
  for (let k = 0; k < iters; k++) {
    const a = draw(), b = draw(); if (a === b) continue
    let v = [n[a * 3 + 1] * n[b * 3 + 2] - n[a * 3 + 2] * n[b * 3 + 1], n[a * 3 + 2] * n[b * 3] - n[a * 3] * n[b * 3 + 2], n[a * 3] * n[b * 3 + 1] - n[a * 3 + 1] * n[b * 3]]
    const l = Math.hypot(...v); if (l < 0.05) continue; v = v.map((x) => x / l); if (v[1] < 0) v = v.map((x) => -x); if (v[1] < cosMax) continue
    const [s] = cons(v, tol); if (s > bs) { bs = s; best = v }
  }
  if (!best) return null
  /* 다듬기: 최소제곱은 허용 폭 안의 잡동사니에 끌려요 → 합의(맞는 무게)를 직접 최대로 하는 촘촘한 국소 탐색 (±1.6°, 0.1° 간격, 좁은 허용 0.008) */
  { const e1 = [1, 0, 0], e2 = [0, 0, 1], rot = (v, a, b) => { const x = v[0] + a * e1[0] + b * e2[0], y = v[1] + a * e1[1] + b * e2[1], z = v[2] + a * e1[2] + b * e2[2], l = Math.hypot(x, y, z); return [x / l, y / l, z / l] }
    for (const [span, stp] of [[0.028, 0.00175], [0.004, 0.0004]]) { let bb = best, bsc = cons(best, tol * 0.67)[0]; for (let a = -span; a <= span + 1e-9; a += stp) for (let b = -span; b <= span + 1e-9; b += stp) { const v = rot(best, a, b), sc = cons(v, tol * 0.67)[0]; if (sc > bsc) { bsc = sc; bb = v } } best = bb } }
  for (const t of [tol * 0.67]) { const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; for (const i of sel) { const a = n[i * 3], b = n[i * 3 + 1], c = n[i * 3 + 2]; if (Math.abs(a * best[0] + b * best[1] + c * best[2]) > t) continue; const ww = w[i]; S[0][0] += ww * a * a; S[0][1] += ww * a * b; S[0][2] += ww * a * c; S[1][1] += ww * b * b; S[1][2] += ww * b * c; S[2][2] += ww * c * c } S[1][0] = S[0][1]; S[2][0] = S[0][2]; S[2][1] = S[1][2]; const e = smallestEig(S); best = e[1] < 0 ? e.map((x) => -x) : e }
  const [s, c] = cons(best, tol); let tot = 0; for (const i of sel) tot += w[i]
  return { v: best, inl: c, frac: s / tot, tilt: (Math.acos(Math.min(1, best[1])) * 180) / Math.PI }
}
/* 믿음 규칙 (calib-level 단계가 실제 파노라마 + 아는 기울기로 고른 값을 여기 적어요) */
export const LEVEL_RULE = { minInl: 120, agree: 1.0, rel: 0.15, maxApply: 25 }
export function upParts(E, { tol = 0.012 } = {}) { const k = E.w.length, all = [], left = [], right = []; for (let i = 0; i < k; i++) { all.push(i); (E.az[i] < 0.5 ? left : right).push(i) } return { g: ransacUp(E, all, { tol }), a: ransacUp(E, left, { seed: 11, tol }), b: ransacUp(E, right, { seed: 13, tol }) } }
export function trustUp({ g, a, b }, R = LEVEL_RULE) { if (!g || !a || !b) return { ok: false, ang: 99 }; const ang = (Math.acos(Math.min(1, Math.abs(a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2]))) * 180) / Math.PI; return { ok: a.inl >= R.minInl && b.inl >= R.minInl && ang <= Math.max(R.agree, R.rel * g.tilt) && g.tilt <= R.maxApply, ang } }
export function upVector2(E, R = LEVEL_RULE) {
  const { g, a, b } = upParts(E)
  if (!g) return { v: [0, 1, 0], tilt: 0, conf: 0, ok: false, why: '경계가 적어요' }
  const t = trustUp({ g, a, b }, R)
  return { v: g.v, tilt: g.tilt, conf: g.frac, ok: t.ok, halves: +t.ang.toFixed(2), inl: g.inl }
}
// 맨해튼 점수: 수평을 맞춘 뒤(위 = v) 세로가 아닌 경계들 중, 수평면의 직교 소실점 쌍 (h, h⊥) 의 큰 원에 맞는 무게 비율 (최댓값)
export function manhattanScore(E, v = [0, 1, 0], { tol = 0.035, vert = 0.12 } = {}) {
  const { n, w } = E, k = w.length, ax = Math.abs(v[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]
  let e1 = [v[1] * ax[2] - v[2] * ax[1], v[2] * ax[0] - v[0] * ax[2], v[0] * ax[1] - v[1] * ax[0]]; const l1 = Math.hypot(...e1); e1 = e1.map((x) => x / l1)
  const e2 = [v[1] * e1[2] - v[2] * e1[1], v[2] * e1[0] - v[0] * e1[2], v[0] * e1[1] - v[1] * e1[0]]
  const idx = []; let tot = 0; for (let i = 0; i < k; i++) { const nv = n[i * 3] * v[0] + n[i * 3 + 1] * v[1] + n[i * 3 + 2] * v[2]; if (Math.abs(nv) < vert) continue; idx.push(i); tot += w[i] } /* 세로선(n ⟂ v)은 빼고 */
  if (tot <= 0) return { score: 0, tot: 0, phi: 0 }
  let best = 0, bphi = 0
  for (let s = 0; s < 90; s++) {
    const phi = (s * Math.PI) / 180, h1 = [0, 1, 2].map((j) => Math.cos(phi) * e1[j] + Math.sin(phi) * e2[j]), h2 = [0, 1, 2].map((j) => -Math.sin(phi) * e1[j] + Math.cos(phi) * e2[j]); let sc = 0
    for (const i of idx) { const a = n[i * 3], b = n[i * 3 + 1], c = n[i * 3 + 2]; if (Math.abs(a * h1[0] + b * h1[1] + c * h1[2]) < tol || Math.abs(a * h2[0] + b * h2[1] + c * h2[2]) < tol) sc += w[i] }
    if (sc > best) { best = sc; bphi = phi }
  }
  return { score: best / tot, tot, phi: bphi }
}
// v 를 위(0,1,0)로 돌리는 회전 R (로드리게스) — 3×3 행 우선
export function levelRotation(v) {
  const up = [0, 1, 0], ax = [v[1] * up[2] - v[2] * up[1], v[2] * up[0] - v[0] * up[2], v[0] * up[1] - v[1] * up[0]], s = Math.hypot(...ax), c = v[0] * up[0] + v[1] * up[1] + v[2] * up[2]
  if (s < 1e-9) return [1, 0, 0, 0, 1, 0, 0, 0, 1]
  const k = ax.map((x) => x / s), K = [0, -k[2], k[1], k[2], 0, -k[0], -k[1], k[0], 0], K2 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let m = 0; m < 3; m++) K2[i * 3 + j] += K[i * 3 + m] * K[m * 3 + j]
  return [0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => (i % 4 === 0 ? 1 : 0) + s * K[i] + (1 - c) * K2[i])
}
// 등장방형 회전: 결과 화소 방향 d → 원본 방향 Rᵀd 에서 쌍선형 샘플 (ch 채널)
export function rotateEquirect(src, W, H, R, ch = 3) {
  const out = new Uint8Array(W * H * ch), d = [0, 0, 0]
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    dirOf((x + 0.5) / W, (y + 0.5) / H, d)
    const sx = R[0] * d[0] + R[3] * d[1] + R[6] * d[2], sy = R[1] * d[0] + R[4] * d[1] + R[7] * d[2], sz = R[2] * d[0] + R[5] * d[1] + R[8] * d[2]
    const lon = Math.atan2(sx, -sz), lat = Math.asin(Math.max(-1, Math.min(1, sy))), fx = (lon / (2 * Math.PI) + 0.5) * W - 0.5, fy = Math.max(0, Math.min(H - 1.001, (0.5 - lat / Math.PI) * H - 0.5))
    const x0 = Math.floor(fx), y0 = Math.floor(fy), ax = fx - x0, ay = fy - y0, X0 = ((x0 % W) + W) % W, X1 = (X0 + 1) % W, Y1 = Math.min(H - 1, y0 + 1), o = (y * W + x) * ch
    for (let c = 0; c < ch; c++) out[o + c] = Math.round((src[(y0 * W + X0) * ch + c] * (1 - ax) + src[(y0 * W + X1) * ch + c] * ax) * (1 - ay) + (src[(Y1 * W + X0) * ch + c] * (1 - ax) + src[(Y1 * W + X1) * ch + c] * ax) * ay)
  }
  return out
}
// 한 번에: 판정 + (필요하면) 수평 맞추기. 1024×512 사본으로 재고, 원본 크기에 같은 회전을 적용해요
export function analyzePano(rgb, W, H) {
  const E = edgeNormals(rgb, W, H), up = upVector2(E), man = up.ok ? manhattanScore(E, up.v) : { score: null, tot: 0 }
  return { tilt: +up.tilt.toFixed(2), conf: +up.conf.toFixed(3), v: up.v.map((x) => +x.toFixed(5)), level: up.ok && up.tilt > 0.4, halves: up.halves, manh: man.score == null ? null : +man.score.toFixed(3), edges: E.w.length }
}
/* 3판: 깊이로 수평 재기 (보정 결과: 경계 방식은 실제 파노라마에서 반듯한 사진도 중앙 3.8° 틀려 믿을 수 있는 경우가 0.3% 뿐).
   병합 깊이 g ≈ S/r + T (아핀 역깊이). 지면은 평면: 방향 d 의 지면 점은 1/r = −(n·d)/h (n = 세계의 위, 카메라 높이 h)
   → g = w·d + T, w = −(S/h)·n. 아래 반구 화소로 4개 수 (w, T) 를 RANSAC + 최소제곱 → n = −w/|w|, S = |w|·h.
   반듯하다고 가정한 지금의 지면 맞춤(n = (0,1,0))의 일반화예요. 지면은 사진에서 넓어서 경계보다 훨씬 튼튼해요. */
function solve4(M, y) { const A = M.map((r, i) => [...r, y[i]]); for (let c = 0; c < 4; c++) { let p = c; for (let r = c + 1; r < 4; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r; if (Math.abs(A[p][c]) < 1e-12) return null; [A[c], A[p]] = [A[p], A[c]]; for (let r = 0; r < 4; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let k = c; k < 5; k++) A[r][k] -= f * A[c][k] } } return A.map((r, i) => r[4] / r[i]) }
export function groundUp(g, W, H, { h = 1.6, latMin = 3, latMax = 80, step = 4, tolK = 0.03, iters = 500, seed = 5, maxTilt = 20 } = {}) {
  const P = [], Y = [], d = [0, 0, 0]
  for (let y = 0; y < H; y += step) { const lat = (0.5 - (y + 0.5) / H) * 180; if (lat > -latMin || lat < -latMax) continue; for (let x = 0; x < W; x += step) { const v = g[y * W + x]; if (!Number.isFinite(v)) continue; dirOf((x + 0.5) / W, (y + 0.5) / H, d); P.push(d[0], d[1], d[2]); Y.push(v) } }
  const n = Y.length; if (n < 50) return null
  const med = Y.map(Math.abs).sort((a, b) => a - b)[n >> 1], tol = Math.max(1e-6, med * tolK), R = rnd(seed), cosMax = Math.cos((maxTilt * Math.PI) / 180)
  const fit = (idx) => { const M = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], b = [0, 0, 0, 0]; for (const i of idx) { const r = [P[i * 3], P[i * 3 + 1], P[i * 3 + 2], 1]; for (let a = 0; a < 4; a++) { b[a] += r[a] * Y[i]; for (let c = 0; c < 4; c++) M[a][c] += r[a] * r[c] } } return solve4(M, b) }
  const count = (p) => { let c = 0; for (let i = 0; i < n; i++) if (Math.abs(p[0] * P[i * 3] + p[1] * P[i * 3 + 1] + p[2] * P[i * 3 + 2] + p[3] - Y[i]) < tol) c++; return c }
  const upOf = (p) => { const l = Math.hypot(p[0], p[1], p[2]); return l > 0 ? [-p[0] / l, -p[1] / l, -p[2] / l] : null }
  let best = null, bc = -1
  for (let k = 0; k < iters; k++) {
    const idx = [0, 1, 2, 3].map(() => Math.floor(R() * n)); const M = idx.map((i) => [P[i * 3], P[i * 3 + 1], P[i * 3 + 2], 1]), p = solve4(M, idx.map((i) => Y[i])); if (!p) continue
    const u = upOf(p); if (!u || u[1] < cosMax) continue /* 위가 위쪽 20° 안 · 지면은 카메라보다 아래 */
    const c = count(p); if (c > bc) { bc = c; best = p }
  }
  if (!best) return null
  for (let it = 0; it < 2; it++) { const idx = []; for (let i = 0; i < n; i++) if (Math.abs(best[0] * P[i * 3] + best[1] * P[i * 3 + 1] + best[2] * P[i * 3 + 2] + best[3] - Y[i]) < tol) idx.push(i); const p = fit(idx); if (p && upOf(p) && upOf(p)[1] >= cosMax) best = p }
  const up = upOf(best), inl = count(best) / n, S = Math.hypot(best[0], best[1], best[2]) * h
  return { up, tilt: (Math.acos(Math.min(1, up[1])) * 180) / Math.PI, inl, S, T: best[3] }
}
/* 깊이 지도(실수) 회전 — rotateEquirect 와 같은 규칙, 쌍선형 (무한대는 가장 가까운 값) */
export function rotateEquirectF(src, W, H, R) {
  const out = new Float32Array(W * H), d = [0, 0, 0]
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    dirOf((x + 0.5) / W, (y + 0.5) / H, d)
    const sx = R[0] * d[0] + R[3] * d[1] + R[6] * d[2], sy = R[1] * d[0] + R[4] * d[1] + R[7] * d[2], sz = R[2] * d[0] + R[5] * d[1] + R[8] * d[2]
    const lon = Math.atan2(sx, -sz), lat = Math.asin(Math.max(-1, Math.min(1, sy))), fx = (lon / (2 * Math.PI) + 0.5) * W - 0.5, fy = Math.max(0, Math.min(H - 1.001, (0.5 - lat / Math.PI) * H - 0.5))
    const x0 = Math.floor(fx), y0 = Math.floor(fy), ax = fx - x0, ay = fy - y0, X0 = ((x0 % W) + W) % W, X1 = (X0 + 1) % W, Y1 = Math.min(H - 1, y0 + 1)
    const v = [src[y0 * W + X0], src[y0 * W + X1], src[Y1 * W + X0], src[Y1 * W + X1]]
    out[y * W + x] = v.every(Number.isFinite) ? (v[0] * (1 - ax) + v[1] * ax) * (1 - ay) + (v[2] * (1 - ax) + v[3] * ax) * ay : v[(ay > 0.5 ? 2 : 0) + (ax > 0.5 ? 1 : 0)]
  }
  return out
}
