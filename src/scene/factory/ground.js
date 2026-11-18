// v1.0-C 지면 캘리브레이션 — (sin φ, d) 공간 RANSAC 직선 맞춤 (설계 근거서 4-6)
//  · 지면 위 점은 카메라 높이 h 에서 1/r = sin φ / h (파노라마) · 1/Z = (y − y_h)/(f·h) (원근 사진).
//  · 아핀 역깊이 d = S·(1/r) + T 이므로 지면 화소의 (sin φ, d) 는 직선: 기울기 = S/h, 절편 = T.
//    3D 로 먼저 역투영하면 모르는 T 때문에 평면이 휘어요 → 그래서 이 공간에서 직선을 맞춰요.
//  · 차 · 사람 · 건물 밑동은 직선에서 벗어난 점(이상치)이라 RANSAC 이 걸러요.
import { rng } from '../geom.js'

// 일반 RANSAC 직선: xs, ys → { m, c, inl(비율), n }. tol: 절대 허용 오차
export function ransacLine(xs, ys, { iters = 400, tol, seed = 3 } = {}) {
  const n = xs.length, R = rng(seed); if (n < 2) return null
  let best = null, bi = -1
  for (let k = 0; k < iters; k++) {
    const i = (R() * n) | 0, j = (R() * n) | 0; if (i === j || Math.abs(xs[i] - xs[j]) < 1e-3) continue
    const m = (ys[j] - ys[i]) / (xs[j] - xs[i]), c = ys[i] - m * xs[i]; if (m <= 0) continue /* 가까울수록(φ 클수록) 역깊이가 커야 지면 */
    let cnt = 0; for (let q = 0; q < n; q++) if (Math.abs(ys[q] - m * xs[q] - c) < tol) cnt++
    if (cnt > bi) { bi = cnt; best = { m, c } }
  }
  if (!best) return null
  for (let it = 0; it < 3; it++) { /* 안쪽 점으로 최소제곱 다듬기 */
    let sx = 0, sy = 0, sxx = 0, sxy = 0, k = 0
    for (let q = 0; q < n; q++) if (Math.abs(ys[q] - best.m * xs[q] - best.c) < tol) { sx += xs[q]; sy += ys[q]; sxx += xs[q] * xs[q]; sxy += xs[q] * ys[q]; k++ }
    const den = k * sxx - sx * sx; if (k < 2 || Math.abs(den) < 1e-12) break; best = { m: (k * sxy - sx * sy) / den, c: (sy - ((k * sxy - sx * sy) / den) * sx) / k }; bi = k
  }
  return { ...best, inl: bi / n, n }
}
// 파노라마: 수평선 아래(φ ∈ [φ0, φ1]) 화소를 성기게 모아 맞춰요 → { S, T, h, inl }. 거리 r = S / (d − T)
export function fitGroundPano(g, W, H, { h = 1.6, phi0 = 4, phi1 = 75, step = 3, tolK = 0.03 } = {}) {
  const xs = [], ys = []
  for (let y = 0; y < H; y += step) { const lat = (0.5 - (y + 0.5) / H) * Math.PI, phi = -lat * 180 / Math.PI; if (phi < phi0 || phi > phi1) continue; for (let x = 0; x < W; x += step) { xs.push(Math.sin(-lat)); ys.push(g[y * W + x]) } }
  const srt = ys.map(Math.abs).sort((a, b) => a - b), tol = Math.max(1e-6, srt[srt.length >> 1] * tolK), L = ransacLine(xs, ys, { tol })
  if (!L) return null
  return { S: L.m * h, T: L.c, h, inl: L.inl, slope: L.m }
}
// 원근 사진: 화소 행 y 와 d (수평선 y_h, 초점 f 화소) → 같은 직선 맞춤. x = (y − y_h)/f
export function fitGroundPersp(rows, ds, yh, f, { h = 1.6, tolK = 0.03 } = {}) {
  const xs = [], ys = []; for (let i = 0; i < rows.length; i++) { const x = (rows[i] - yh) / f; if (x > 0.02) { xs.push(x); ys.push(ds[i]) } }
  const srt = ys.map(Math.abs).sort((a, b) => a - b), L = ransacLine(xs, ys, { tol: Math.max(1e-6, srt[srt.length >> 1] * tolK) }); return L && { S: L.m * h, T: L.c, h, inl: L.inl }
}
// d → 거리 r (m). 하늘(d ≤ T 근처)은 Infinity
export function toMetric(g, cal, { far = 400 } = {}) { const r = new Float32Array(g.length); for (let i = 0; i < g.length; i++) { const q = g[i] - cal.T; r[i] = q > cal.S / far ? cal.S / q : Infinity } return r }
// 축척 합치기 (가중 최소제곱, 로그 공간): 지면(h 가정) · 알려진 물체 크기(차 폭 1.8m 등)에서 나온 S 추정들
//  · 물체: 각폭 α(라디안) · 그 물체의 역깊이 d → r = W/(2·tan(α/2)) 와 r = S/(d − T) 가 같아야 → S = r·(d − T)
export function objectScale({ widthM = 1.8, alpha, d, T }) { const r = widthM / (2 * Math.tan(alpha / 2)); return r * (d - T) }
export function fuseScale(list) { let s = 0, w = 0; for (const e of list) if (e && e.S > 0 && e.w > 0) { s += e.w * Math.log(e.S); w += e.w } return w ? Math.exp(s / w) : null }
