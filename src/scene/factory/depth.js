// v1.0-C 깊이: 큐브맵 6면 → 면마다 아핀 역깊이 → 면 사이 최소제곱 정렬 → 등장방형 하나로
//  · 깊이 모델(Depth Anything V2 Small)은 원근 사진 한 장마다 d = s/Z + t (Z = 광축 방향 깊이) 를 내요. 면마다 s, t 가 달라요.
//  · 같은 점의 거리 r 와 면 i 의 Z 는 Z = r·cos θᵢ (θᵢ = 광축과의 각). 그래서 1/r = cos θᵢ · (aᵢ·dᵢ + bᵢ), aᵢ = 1/sᵢ, bᵢ = −tᵢ/sᵢ
//    → 겹치는 띠의 같은 방향에서 cos θᵢ(aᵢdᵢ + bᵢ) − cos θⱼ(aⱼdⱼ + bⱼ) = 0 이 (a, b) 에 대해 1차식이에요. 12개 미지수의 최소제곱.
//  · 면 안에서 cos θ 가 0.6 ~ 1 로 변해서 시프트 b 도 정해져요(축척 a 하나만 기준 면에서 1 로 고정). 남은 전체 축척 · 시프트는 지면 캘리브레이션이 정해요.
import { faceDir, faceUV, dirOf, uvOf, rng } from '../geom.js'

export const FOV = (100 * Math.PI) / 180 /* 면마다 90° + 여유 10° → 이웃 면과 겹치는 띠 */
const bil = (img, N, x, y) => { x = Math.max(0, Math.min(N - 1.001, x)); y = Math.max(0, Math.min(N - 1.001, y)); const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i = y0 * N + x0; return (img[i] * (1 - fx) + img[i + 1] * fx) * (1 - fy) + (img[i + N] * (1 - fx) + img[i + N + 1] * fx) * fy }
// 등장방형 RGB(Uint8 ×3) → 면 i 의 RGB (깊이 모델 입력)
export function faceRGB(rgb, W, H, i, N, fov = FOV) {
  const out = new Uint8Array(N * N * 3), d = [0, 0, 0], uv = [0, 0]
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    faceDir(i, x, y, N, fov, d); uvOf(d[0], d[1], d[2], uv); const fx = uv[0] * W - 0.5, fy = Math.max(0, Math.min(H - 1.001, uv[1] * H - 0.5)), x0 = Math.floor(fx), y0 = fy | 0, ax = fx - x0, ay = fy - y0
    for (let c = 0; c < 3; c++) { const p = (xx, yy) => rgb[(yy * W + ((xx % W) + W) % W) * 3 + c]; out[(y * N + x) * 3 + c] = Math.round((p(x0, y0) * (1 - ax) + p(x0 + 1, y0) * ax) * (1 - ay) + (p(x0, y0 + 1) * (1 - ax) + p(x0 + 1, y0 + 1) * ax) * ay) }
  }
  return out
}
// 시험용 깊이 모델 흉내: 정답 거리 r(등장방형) → 면마다 무작위 (s, t) 아핀 역깊이 + 잡음. 하늘(∞) 은 d = t
export function simDepthFaces(r, W, H, N, { seed = 1, noise = 0.01, fov = FOV } = {}) {
  const R = rng(seed), faces = [], truth = [], d = [0, 0, 0], uv = [0, 0]
  for (let i = 0; i < 6; i++) {
    const s = 0.5 + R() * 1.5, t = (R() - 0.3) * 0.3, img = new Float32Array(N * N); truth.push({ s, t })
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      faceDir(i, x, y, N, fov, d); uvOf(d[0], d[1], d[2], uv); const ex = Math.min(W - 1, Math.floor(uv[0] * W)), ey = Math.min(H - 1, Math.floor(uv[1] * H)), rr = r[ey * W + ex]
      const [f] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]].slice(i, i + 1), cos = d[0] * f[0] + d[1] * f[1] + d[2] * f[2]
      img[y * N + x] = (Number.isFinite(rr) ? s / (rr * cos) : 0) + t + (R() - 0.5) * 2 * noise * s
    }
    faces.push(img)
  }
  return { faces, truth }
}
// 작은 연립방정식 (정규방정식) — 가우스 소거 (부분 피벗)
export function solve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]])
  for (let c = 0; c < n; c++) { let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r; [M[c], M[p]] = [M[p], M[c]]; const v = M[c][c] || 1e-12; for (let r = 0; r < n; r++) if (r !== c) { const k = M[r][c] / v; if (k) for (let q = c; q <= n; q++) M[r][q] -= k * M[c][q] } }
  return M.map((r, i) => r[n] / (r[i] || 1e-12))
}
// 면 사이 정렬: 각 면의 성긴 격자 표본을 이웃 면에서도 읽어 식 하나씩. ref 면의 a = 1 로 고정 (축척 기준), b 는 아주 약한 릿지(λ)
export function alignFaces(faces, N, { fov = FOV, ref = 5, step = 4, lambda = 1e-6 } = {}) {
  const A = Array.from({ length: 12 }, () => new Array(12).fill(0)), B = new Array(12).fill(0), d = [0, 0, 0]; let rows = 0
  const add = (co, rhs, w) => { for (const [p, x] of co) { for (const [q, y] of co) A[p][q] += w * x * y; B[p] += w * x * rhs } rows++ }
  for (let i = 0; i < 6; i++) for (let y = 0; y < N; y += step) for (let x = 0; x < N; x += step) {
    const ui = faceUV(i, faceDir(i, x, y, N, fov, d), N, fov); if (!ui || ui.edge > 0.12) continue /* 가장자리 띠만 겹쳐요 */
    for (let j = i + 1; j < 6; j++) {
      const uj = faceUV(j, d, N, fov); if (!uj) continue
      const di = bil(faces[i], N, ui.x, ui.y), dj = bil(faces[j], N, uj.x, uj.y), co = [[2 * i, ui.cos * di], [2 * i + 1, ui.cos], [2 * j, -uj.cos * dj], [2 * j + 1, -uj.cos]]
      add(co, 0, 1)
    }
  }
  for (let k = 0; k < 6; k++) A[2 * k + 1][2 * k + 1] += lambda * rows
  const w = 1e6 * rows; A[2 * ref][2 * ref] += w; B[2 * ref] += w /* a_ref = 1 */
  const x = solve(A, B); return { a: [0, 1, 2, 3, 4, 5].map((k) => x[2 * k]), b: [0, 1, 2, 3, 4, 5].map((k) => x[2 * k + 1]), rows }
}
// 정렬된 면들 → 등장방형 g = 1/r 에 비례하는 값 (가장자리로 갈수록 가중치를 줄여 이음새 없이)
export function mergeFaces(faces, ab, W, H, N, { fov = FOV } = {}) {
  const g = new Float32Array(W * H), d = [0, 0, 0]
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    dirOf((x + 0.5) / W, (y + 0.5) / H, d); let s = 0, ws = 0
    for (let i = 0; i < 6; i++) { const u = faceUV(i, d, N, fov); if (!u) continue; const w = Math.max(1e-4, u.edge) ** 2; s += w * u.cos * (ab.a[i] * bil(faces[i], N, u.x, u.y) + ab.b[i]); ws += w }
    g[y * W + x] = s / ws
  }
  return g
}
