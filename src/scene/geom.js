// v1.0-C 장면 기하 (Node 장면 공장 · 브라우저 공용, 의존성 없음)
//  · 등장방형(equirect) 좌표: u = (x + 0.5) / W, v = (y + 0.5) / H, 경도 lon = (u − 0.5)·2π, 위도 lat = (0.5 − v)·π
//  · 방향: x = cos(lat)·sin(lon), y = sin(lat), z = −cos(lat)·cos(lon)  (lon = 0 → −Z, Three.js 카메라의 앞)
//  · 지면 아래 각도 φ = −lat. 카메라 높이 h 의 지면 점까지 거리 r = h / sin φ (φ > 0)
export const TAU = Math.PI * 2
export function dirOf(u, v, out = [0, 0, 0]) {
  const lon = (u - 0.5) * TAU, lat = (0.5 - v) * Math.PI, c = Math.cos(lat)
  out[0] = c * Math.sin(lon); out[1] = Math.sin(lat); out[2] = -c * Math.cos(lon); return out
}
export function uvOf(x, y, z, out = [0, 0]) {
  const lon = Math.atan2(x, -z), lat = Math.asin(Math.max(-1, Math.min(1, y / (Math.hypot(x, y, z) || 1))))
  out[0] = lon / TAU + 0.5; out[1] = 0.5 - lat / Math.PI; return out
}
// 큐브맵 6면: [앞축, 오른쪽, 위]. 순서 +X −X +Y −Y +Z −Z
export const FACES = [
  [[1, 0, 0], [0, 0, 1], [0, 1, 0]], [[-1, 0, 0], [0, 0, -1], [0, 1, 0]],
  [[0, 1, 0], [1, 0, 0], [0, 0, 1]], [[0, -1, 0], [1, 0, 0], [0, 0, -1]],
  [[0, 0, 1], [-1, 0, 0], [0, 1, 0]], [[0, 0, -1], [1, 0, 0], [0, 1, 0]],
]
// 면 i 의 화소 (px, py) (크기 N, 시야각 fov 라디안, 여유를 둬 이웃 면과 겹쳐요) → 방향 · 광축과의 cos
export function faceDir(i, px, py, N, fov, out = [0, 0, 0]) {
  const [f, r, up] = FACES[i], k = Math.tan(fov / 2), a = ((px + 0.5) / N * 2 - 1) * k, b = (1 - (py + 0.5) / N * 2) * k
  const x = f[0] + a * r[0] + b * up[0], y = f[1] + a * r[1] + b * up[1], z = f[2] + a * r[2] + b * up[2], n = Math.hypot(x, y, z)
  out[0] = x / n; out[1] = y / n; out[2] = z / n; return out
}
// 방향 → 면 i 의 화소 좌표 (면 앞쪽이 아니면 null). cos = 광축과의 cos (원근 깊이 Z = r·cos)
export function faceUV(i, d, N, fov) {
  const [f, r, up] = FACES[i], c = d[0] * f[0] + d[1] * f[1] + d[2] * f[2]; if (c <= 1e-6) return null
  const k = Math.tan(fov / 2), a = (d[0] * r[0] + d[1] * r[1] + d[2] * r[2]) / c / k, b = (d[0] * up[0] + d[1] * up[1] + d[2] * up[2]) / c / k
  if (Math.abs(a) > 1 || Math.abs(b) > 1) return null
  return { x: (a + 1) / 2 * N - 0.5, y: (1 - b) / 2 * N - 0.5, cos: c, edge: 1 - Math.max(Math.abs(a), Math.abs(b)) }
}
// 결정적 난수 (장면 공장 · 선택 시험이 매번 같게)
export function rng(seed = 1) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296 } }
