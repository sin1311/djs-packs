// 인서트 컷(보통 사진) 만들기용 순수 함수 — 장면 공장(scene-factory.mjs buildShot)과 단위 시험이 함께 써요
// 얼굴 찾기 타일: YuNet 입력(S×S)에 맞춰 원본 해상도로 겹치게 자른 칸들 + 전체를 줄인 한 칸(큰 얼굴용)
export function faceTiles(W, H, S = 640, overlap = 0.2) {
  const st = Math.round(S * (1 - overlap)), xs = [], ys = [], out = []
  for (let x = 0; ; x += st) { xs.push(Math.min(x, Math.max(0, W - S))); if (x + S >= W) break }
  for (let y = 0; ; y += st) { ys.push(Math.min(y, Math.max(0, H - S))); if (y + S >= H) break }
  for (const y of [...new Set(ys)]) for (const x of [...new Set(xs)]) out.push({ x, y, s: 1, w: Math.min(S, W), h: Math.min(S, H) })
  const k = Math.max(W, H) / S; if (k > 1.01) out.push({ x: 0, y: 0, s: k, w: W, h: H }) /* 전체를 S 안에 줄여서 */
  return out
}
// 가까운 인물 사진: 얼굴 하나가 사진의 maxFrac 넘게 차지하면 컷으로 쓰지 않아요 (흐린 큰 얼굴은 어색하고, 일상 · 분위기 컷의 뜻과도 달라요)
export const closeFace = (boxes, W, H, maxFrac = 0.015) => boxes.some((b) => (b.w * b.h) / (W * H) > maxFrac)
// 겹친 상자 합치기 (타일 경계에서 같은 얼굴이 두 번)
export function mergeBoxes(boxes, thr = 0.3) {
  const iou = (a, b) => { const x = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)), y = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y)), i = x * y; return i / (a.w * a.h + b.w * b.h - i) }
  const keep = []; for (const b of [...boxes].sort((p, q) => (q.score || 0) - (p.score || 0))) if (keep.every((k) => iou(k, b) < thr)) keep.push(b); return keep
}
// 상대 깊이(가까울수록 큼) → 0–1, 2~98 백분위로 늘려요 (튀는 값에 흔들리지 않게)
export function normDisp(d) {
  const s = Float32Array.from(d).sort(), lo = s[Math.floor(s.length * 0.02)], hi = s[Math.floor(s.length * 0.98)], out = new Float32Array(d.length), r = hi - lo || 1
  for (let i = 0; i < d.length; i++) out[i] = Math.max(0, Math.min(1, (d[i] - lo) / r)); return out
}
// 차 사진의 번호판: 사진 아래쪽 85% 안 · 폭이 사진의 1.5~12% · 점수 순 3개까지 (원근 사진이라 거리 · 높이 대신 크기로)
export function shotPlates(cands, W, H, { maxN = 3 } = {}) {
  return cands.filter((b) => b.y + b.h / 2 > H * 0.15 && b.w >= W * 0.015 && b.w <= W * 0.12).sort((a, b) => (b.score || 0) - (a.score || 0)).slice(0, maxN)
}
