// v1.0-C 얼굴 · 번호판 흐림 (거리 사진은 필수 — 불변 조건)
//  · 흐림: 상자마다 타원 마스크(가장자리 부드럽게) 안을 강한 가우스로. 원본 화소가 남지 않게 두 번 흐리고 블록 평균도 섞어요(되돌리기 어렵게).
//  · 찾기: Actions 에서 YuNet(얼굴, OpenCV Zoo · MIT) ONNX. 번호판은 아래 휴리스틱 후보(밝은 바탕 · 세로 경계가 촘촘한 가로 직사각형)를 흐리고,
//    보고서에 "사람이 확인할 목록"으로 남겨요. 후보 검출은 완벽하지 않아서, CC-BY 거리 사진은 사람 확인 전에는 팩에 넣지 않아요(공장 규칙).
export function blurBoxes(rgb, W, H, boxes, { ch = 3 } = {}) {
  const out = rgb.slice()
  for (const b of boxes) {
    const pad = Math.ceil(Math.max(b.w, b.h) * 0.25), x0 = Math.floor(b.x - pad), y0 = Math.max(0, Math.floor(b.y - pad)), x1 = Math.ceil(b.x + b.w + pad), y1 = Math.min(H, Math.ceil(b.y + b.h + pad)), bw = x1 - x0, bh = y1 - y0
    if (bw < 2 || bh < 2) continue
    const cell = Math.max(2, Math.round(Math.min(b.w, b.h) / 4)), sig = Math.max(2, Math.min(b.w, b.h) / 3), buf = new Float32Array(bw * bh * ch)
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) { const xx = (((x0 + x) % W) + W) % W, i = ((y0 + y) * W + xx) * ch; for (let c = 0; c < ch; c++) buf[(y * bw + x) * ch + c] = rgb[i + c] }
    const blk = buf.slice(); for (let y = 0; y < bh; y += cell) for (let x = 0; x < bw; x += cell) for (let c = 0; c < ch; c++) { let s = 0, n = 0; for (let j = y; j < Math.min(bh, y + cell); j++) for (let i = x; i < Math.min(bw, x + cell); i++) { s += buf[(j * bw + i) * ch + c]; n++ } for (let j = y; j < Math.min(bh, y + cell); j++) for (let i = x; i < Math.min(bw, x + cell); i++) blk[(j * bw + i) * ch + c] = s / n }
    let cur = blk; for (let rep = 0; rep < 2; rep++) cur = gauss(cur, bw, bh, ch, sig)
    const cx = b.x + b.w / 2 - x0, cy = b.y + b.h / 2 - y0, rx = b.w / 2 + pad * 0.5, ry = b.h / 2 + pad * 0.5
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      const e = Math.hypot((x - cx) / rx, (y - cy) / ry), m = e < 0.85 ? 1 : e > 1.15 ? 0 : (1.15 - e) / 0.3; if (m <= 0) continue
      const xx = (((x0 + x) % W) + W) % W, i = ((y0 + y) * W + xx) * ch; for (let c = 0; c < ch; c++) out[i + c] = Math.round(rgb[i + c] * (1 - m) + cur[(y * bw + x) * ch + c] * m)
    }
  }
  return out
}
function gauss(src, w, h, ch, s) {
  const r = Math.ceil(s * 2.5), k = []; let ks = 0; for (let i = -r; i <= r; i++) { const v = Math.exp(-(i * i) / (2 * s * s)); k.push(v); ks += v }
  const t = new Float32Array(src.length), o = new Float32Array(src.length)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < ch; c++) { let a = 0; for (let i = -r; i <= r; i++) a += src[(y * w + Math.max(0, Math.min(w - 1, x + i))) * ch + c] * k[i + r]; t[(y * w + x) * ch + c] = a / ks }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < ch; c++) { let a = 0; for (let i = -r; i <= r; i++) a += t[(Math.max(0, Math.min(h - 1, y + i)) * w + x) * ch + c] * k[i + r]; o[(y * w + x) * ch + c] = a / ks }
  return o
}
// 번호판 후보 (휴리스틱): 회색조에서 세로 경계 밀도가 높고, 가로:세로 2~6, 바탕이 밝은 작은 직사각형
export function plateCandidates(rgb, W, H, { minW = 12, maxW = 160 } = {}) {
  const g = new Float32Array(W * H); for (let i = 0; i < W * H; i++) g[i] = 0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2]
  const ev = new Uint8Array(W * H); for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) ev[y * W + x] = Math.abs(g[y * W + x + 1] - g[y * W + x - 1]) > 60 ? 1 : 0
  const I = new Float32Array((W + 1) * (H + 1)); for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) I[(y + 1) * (W + 1) + x + 1] = ev[y * W + x] + I[y * (W + 1) + x + 1] + I[(y + 1) * (W + 1) + x] - I[y * (W + 1) + x]
  const sum = (x, y, w, h) => I[(y + h) * (W + 1) + x + w] - I[y * (W + 1) + x + w] - I[(y + h) * (W + 1) + x] + I[y * (W + 1) + x]
  const out = []
  for (let w = minW; w <= maxW; w = Math.ceil(w * 1.35)) { const h = Math.max(4, Math.round(w / 3.5)), st = Math.max(2, w >> 2); for (let y = 0; y + h < H; y += st) for (let x = 0; x + w < W; x += st) { const den = sum(x, y, w, h) / (w * h); if (den < 0.18 || den > 0.6) continue; let bg = 0; for (let j = y; j < y + h; j += 2) for (let i = x; i < x + w; i += 2) bg += g[j * W + i]; bg /= Math.ceil(h / 2) * Math.ceil(w / 2); if (bg > 110) out.push({ x, y, w, h, score: den }) } }
  out.sort((a, b) => b.score - a.score); const keep = []
  for (const b of out) if (!keep.some((k) => !(b.x > k.x + k.w || k.x > b.x + b.w || b.y > k.y + k.h || k.y > b.y + b.h))) { keep.push(b); if (keep.length >= 40) break }
  return keep
}
// YuNet (face_detection_yunet_2023mar) 출력 풀이: 보폭 8/16/32 마다 cls · obj · bbox · kps. 입력 크기 S(정사각) 기준 화소 상자
export function yunetDecode(out, S, { thr = 0.6 } = {}) {
  const boxes = []
  for (const st of [8, 16, 32]) {
    const cls = out['cls_' + st], obj = out['obj_' + st], bb = out['bbox_' + st]; if (!cls || !obj || !bb) continue
    const n = S / st
    for (let i = 0; i < n * n; i++) {
      const sc = Math.sqrt(Math.max(0, Math.min(1, cls.data[i])) * Math.max(0, Math.min(1, obj.data[i]))); if (sc < thr) continue
      const cx = ((i % n) + bb.data[i * 4]) * st, cy = (Math.floor(i / n) + bb.data[i * 4 + 1]) * st, w = Math.exp(bb.data[i * 4 + 2]) * st, h = Math.exp(bb.data[i * 4 + 3]) * st
      boxes.push({ x: cx - w / 2, y: cy - h / 2, w, h, score: sc })
    }
  }
  boxes.sort((a, b) => b.score - a.score); const keep = []
  for (const b of boxes) { if (keep.every((k) => iou(k, b) < 0.3)) keep.push(b) }
  return keep
}
const iou = (a, b) => { const x = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)), y = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y)), i = x * y; return i / (a.w * a.h + b.w * b.h - i) }
