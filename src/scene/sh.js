// 구면 조화 함수(SH, 2차 = 9계수) 조도 — 하늘 교체 재조명용 (Ramamoorthi & Hanrahan 2001). 장면 공장 · 브라우저 공용
//  · 등장방형 RGB(선형 0–1) → 계수 L[9][3] (입체각 가중 cos 위도)
//  · 법선 n 의 조도 E(n) = Σ Â_l · L_lm · Y_lm(n),  Â = [π, 2π/3, π/4]
import { dirOf } from './geom.js'
const Y = (d) => { const [x, y, z] = d; return [0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z, 0.546274 * (x * x - y * y)] }
export function shProject(get, W, H, step = 2) {
  const L = Array.from({ length: 9 }, () => [0, 0, 0]), d = [0, 0, 0]; let wsum = 0
  for (let y = 0; y < H; y += step) { const lat = (0.5 - (y + 0.5) / H) * Math.PI, w = Math.cos(lat); for (let x = 0; x < W; x += step) { dirOf((x + 0.5) / W, (y + 0.5) / H, d); const c = get(x, y), yb = Y(d); for (let k = 0; k < 9; k++) for (let ch = 0; ch < 3; ch++) L[k][ch] += c[ch] * yb[k] * w; wsum += w } }
  const k4 = (4 * Math.PI) / wsum; for (const c of L) for (let ch = 0; ch < 3; ch++) c[ch] *= k4
  return L
}
const A = [Math.PI, 2.094395, 2.094395, 2.094395, 0.785398, 0.785398, 0.785398, 0.785398, 0.785398]
export function shIrradiance(L, n) { const yb = Y(n), e = [0, 0, 0]; for (let k = 0; k < 9; k++) for (let ch = 0; ch < 3; ch++) e[ch] += A[k] * L[k][ch] * yb[k]; return e.map((v) => Math.max(0, v)) }
// sRGB 8비트 → 선형
export const lin = (v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4) }
// 셰이더용: 9×3 → Float32Array(27) (GLSL vec3 uSH[9])
export const shFlat = (L) => Float32Array.from(L.flat())
