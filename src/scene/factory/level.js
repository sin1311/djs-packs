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
  const N = [], Wt = [], d = [0, 0, 0], a = [0, 0, 0], b = [0, 0, 0], c = [0, 0, 0], e = [0, 0, 0]
  const y0 = Math.ceil(H * (0.5 - latMax / 180)), y1 = Math.floor(H * (0.5 + latMax / 180))
  for (let y = Math.max(1, y0); y < Math.min(H - 1, y1); y += stride) for (let x = 0; x < W; x += stride) {
    const xl = (x - 1 + W) % W, xr = (x + 1) % W, G = (xx, yy) => g[yy * W + xx]
    const gx = (G(xr, y - 1) + 2 * G(xr, y) + G(xr, y + 1)) - (G(xl, y - 1) + 2 * G(xl, y) + G(xl, y + 1)), gy = (G(xl, y + 1) + 2 * G(x, y + 1) + G(xr, y + 1)) - (G(xl, y - 1) + 2 * G(x, y - 1) + G(xr, y - 1))
    const m = Math.hypot(gx, gy) / 4; if (m < minMag) continue
    const u = (x + 0.5) / W, v = (y + 0.5) / H; dirOf(u, v, d); dirOf(u + 0.5 / W, v, a); dirOf(u - 0.5 / W, v, b); dirOf(u, v + 0.5 / H, c); dirOf(u, v - 0.5 / H, e)
    const ex = [a[0] - b[0], a[1] - b[1], a[2] - b[2]], ey = [c[0] - e[0], c[1] - e[1], c[2] - e[2]], lx = Math.hypot(...ex), ly = Math.hypot(...ey)
    const tx = -gy / m, ty = gx / m /* 경계 접선 = 기울기에 수직 (화면 좌표) */, t = [0, 1, 2].map((k) => tx * ex[k] / lx + ty * ey[k] / ly)
    const n = [d[1] * t[2] - d[2] * t[1], d[2] * t[0] - d[0] * t[2], d[0] * t[1] - d[1] * t[0]], nl = Math.hypot(...n); if (nl < 1e-6) continue
    N.push(n[0] / nl, n[1] / nl, n[2] / nl); Wt.push(m)
  }
  return { n: Float32Array.from(N), w: Float32Array.from(Wt) }
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
  const E = edgeNormals(rgb, W, H), up = upVector(E), man = up.ok ? manhattanScore(E, up.v) : { score: null, tot: 0 }
  return { tilt: +up.tilt.toFixed(2), conf: +up.conf.toFixed(3), v: up.v.map((x) => +x.toFixed(5)), level: up.ok && up.tilt > 0.4, manh: man.score == null ? null : +man.score.toFixed(3), edges: E.w.length }
}
