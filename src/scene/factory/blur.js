// v1.0-C 얼굴 · 번호판 흐림 (거리 사진은 필수 — 불변 조건)
//  · 흐림: 상자마다 타원 마스크(가장자리 부드럽게) 안을 강한 가우스로. 원본 화소가 남지 않게 두 번 흐리고 블록 평균도 섞어요(되돌리기 어렵게).
//  · 찾기: Actions 에서 YuNet(얼굴, OpenCV Zoo · MIT) ONNX. 번호판은 아래 휴리스틱 후보(밝은 바탕 · 세로 경계가 촘촘한 가로 직사각형)를 흐리고,
//    보고서에 "사람이 확인할 목록"으로 남겨요. 후보 검출은 완벽하지 않아서, CC-BY 거리 사진은 사람 확인 전에는 팩에 넣지 않아요(공장 규칙).
export function blurBoxes(rgb, W, H, boxes, { ch = 3, wrap = true } = {}) { /* wrap: 360° 파노라마는 좌우가 이어져요 · 보통 사진(인서트 컷)은 false */
  const X = (x) => (wrap ? ((x % W) + W) % W : Math.max(0, Math.min(W - 1, x)))
  const out = rgb.slice()
  for (const b of boxes) {
    const pad = Math.ceil(Math.max(b.w, b.h) * 0.25), x0 = Math.floor(b.x - pad), y0 = Math.max(0, Math.floor(b.y - pad)), x1 = Math.ceil(b.x + b.w + pad), y1 = Math.min(H, Math.ceil(b.y + b.h + pad)), bw = x1 - x0, bh = y1 - y0
    if (bw < 2 || bh < 2) continue
    const cell = Math.max(2, Math.round(Math.min(b.w, b.h) / 4)), sig = Math.max(2, Math.min(b.w, b.h) / 3), buf = new Float32Array(bw * bh * ch)
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) { const xx = X(x0 + x), i = ((y0 + y) * W + xx) * ch; for (let c = 0; c < ch; c++) buf[(y * bw + x) * ch + c] = rgb[i + c] }
    const blk = buf.slice(); for (let y = 0; y < bh; y += cell) for (let x = 0; x < bw; x += cell) for (let c = 0; c < ch; c++) { let s = 0, n = 0; for (let j = y; j < Math.min(bh, y + cell); j++) for (let i = x; i < Math.min(bw, x + cell); i++) { s += buf[(j * bw + i) * ch + c]; n++ } for (let j = y; j < Math.min(bh, y + cell); j++) for (let i = x; i < Math.min(bw, x + cell); i++) blk[(j * bw + i) * ch + c] = s / n }
    let cur = blk; for (let rep = 0; rep < 2; rep++) cur = gauss(cur, bw, bh, ch, sig)
    const cx = b.x + b.w / 2 - x0, cy = b.y + b.h / 2 - y0, rx = b.w / 2 + pad * 0.5, ry = b.h / 2 + pad * 0.5
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      const e = Math.hypot((x - cx) / rx, (y - cy) / ry), m = e < 0.85 ? 1 : e > 1.15 ? 0 : (1.15 - e) / 0.3; if (m <= 0) continue
      const xx = X(x0 + x), i = ((y0 + y) * W + xx) * ch; for (let c = 0; c < ch; c++) out[i + c] = Math.round(rgb[i + c] * (1 - m) + cur[(y * bw + x) * ch + c] * m)
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
/* 번호판 후보 거르기 (깊이를 안 뒤): 번호판은 차에 붙어 지면에서 0.15~1.5 m 높이 · 8 m 안 · 폭 0.3~0.7 m 예요.
   높이 = 눈높이 h + 거리 r·sin(고도각), 실제 폭 = r·(상자 폭 각도). 스튜디오 · 작업장의 무늬(높이 · 폭이 안 맞음)는 걸러져요.
   그래도 한 장면에서 많이 남으면(> maxN) 무늬로 보고 점수 순 maxN 개만 흐려요 (실제 거리에서도 읽을 수 있는 번호판은 몇 개뿐) */
export function plateKeep(cands, r, W, H, { h = 1.6, maxR = 8, lo = 0.15, hi = 1.5, wMin = 0.18, wMax = 0.9, maxN = 8 } = {}) {
  const out = []
  for (const b of cands) {
    const cx = Math.round(((b.x + b.w / 2) % W + W) % W), cy = Math.max(0, Math.min(H - 1, Math.round(b.y + b.h / 2))), rr = r[cy * W + cx]
    if (!Number.isFinite(rr) || rr >= maxR) continue
    const el = (0.5 - (cy + 0.5) / H) * Math.PI, y = h + rr * Math.sin(el), wm = rr * Math.cos(el) * (b.w / W) * 2 * Math.PI
    if (y < lo || y > hi || wm < wMin || wm > wMax) continue
    out.push(b)
  }
  out.sort((a, b) => (b.score || 0) - (a.score || 0)); return out.slice(0, maxN)
}
/* ---- 사생활 2판 (2026-11-19: "모자이크가 너무 많아 그림을 알아보기 힘들어요") ----
   · 얼굴: 누구인지 알아보는 정보는 얼굴 폭당 대략 8~16 주기의 공간 주파수에 있어요. 가우스 σ = 얼굴 폭/8 이면
     대비 10% 경계가 얼굴 폭당 약 2.7 주기로 내려가 신원 정보는 사라지고, 얼굴 모양에 딱 맞는 타원만 바뀌어요
     (예전: 상자 1.4배 + 여백 25% + 블록 평균 + 가우스 두 번 → 주변까지 크게 뭉갰어요).
     흐린 뒤 작은 잡음(±2.5)을 더해 역합성곱(되살리기) 공격을 막아요 — 블록 모자이크는 오히려 되살리기에 약해요.
   · 번호판: 흐리는 대신 "빈 판"으로 — 상자 바로 바깥 테두리 화소로 쿤스 패치(초한 보간)를 채워요
     P(s,t) = (1−t)·위(s) + t·아래(s) + (1−s)·왼(t) + s·오른(t) − 네 귀퉁이 쌍선형. 글자 정보는 0, 조명 기울기는 그대로. */
const hashN = (x, y) => { let h = (x * 374761393 + y * 668265263) | 0; h = (h ^ (h >>> 13)) * 1274126177; return (((h ^ (h >>> 16)) >>> 0) / 4294967296 - 0.5) * 5 }
export function anonBlur(rgb, W, H, boxes, { ch = 3, wrap = true, div = 8, noise = true } = {}) { /* div 8 (2026-11-19 사용자 결정: 6 → 8, 25% 완화). 10% 대비 경계 0.34·w/σ = 얼굴 폭당 약 2.7 주기 — 알아보려면 4~8 주기가 필요해요. 안전한 최대는 10 (3.4 주기) */
  const out = rgb.slice(), X = (x) => (wrap ? ((x % W) + W) % W : Math.max(0, Math.min(W - 1, x)))
  for (const b of boxes) {
    const sig = Math.max(1, Math.min(b.w, b.h) / (b.div || div)), /* b.div: 상자마다 정한 흐림 (minimalDiv — 알아볼 수 없을 만큼만) */ pad = Math.ceil(sig * 3), x0 = Math.floor(b.x - pad), y0 = Math.max(0, Math.floor(b.y - pad)), x1 = Math.ceil(b.x + b.w + pad), y1 = Math.min(H, Math.ceil(b.y + b.h + pad)), bw = x1 - x0, bh = y1 - y0
    if (bw < 2 || bh < 2) continue
    const buf = new Float32Array(bw * bh * ch); for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) { const i = ((y0 + y) * W + X(x0 + x)) * ch; for (let c = 0; c < ch; c++) buf[(y * bw + x) * ch + c] = rgb[i + c] }
    const bl = gauss(buf, bw, bh, ch, sig), cx = b.x + b.w / 2 - x0, cy = b.y + b.h / 2 - y0, rx = b.w / 2, ry = b.h / 2
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      const e = Math.hypot((x - cx) / rx, (y - cy) / ry), m = e < 0.9 ? 1 : e > 1.1 ? 0 : (1.1 - e) / 0.2; if (m <= 0) continue
      const xx = X(x0 + x), i = ((y0 + y) * W + xx) * ch, nz = noise ? hashN(xx, y0 + y) : 0
      for (let c = 0; c < ch; c++) out[i + c] = Math.max(0, Math.min(255, Math.round(rgb[i + c] * (1 - m) + (bl[(y * bw + x) * ch + c] + nz) * m)))
    }
  }
  return out
}
export function plateBlank(rgb, W, H, boxes, { ch = 3, wrap = true, grow = 0.12 } = {}) {
  const out = rgb.slice(), X = (x) => (wrap ? ((x % W) + W) % W : Math.max(0, Math.min(W - 1, x)))
  for (const b0 of boxes) {
    const gx = b0.w * grow, gy = b0.h * grow * 2, x0 = Math.floor(b0.x - gx), x1 = Math.ceil(b0.x + b0.w + gx), y0 = Math.max(1, Math.floor(b0.y - gy)), y1 = Math.min(H - 2, Math.ceil(b0.y + b0.h + gy)), bw = x1 - x0, bh = y1 - y0
    if (bw < 3 || bh < 3) continue
    const px = (x, y, c) => rgb[(y * W + X(x)) * ch + c] /* 원본 테두리 화소 (상자 바로 바깥) */
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const s = (x - x0) / bw, t = (y - y0) / bh, edge = Math.min(x - x0, x1 - x, y - y0, y1 - y), m = Math.min(1, (edge + 0.5) / 1.5), i = (y * W + X(x)) * ch, nz = hashN(X(x), y) * 0.6
      for (let c = 0; c < ch; c++) {
        const top = px(x, y0 - 1, c), bot = px(x, y1 + 1, c), lef = px(x0 - 1, y, c), rig = px(x1 + 1, y, c)
        const P = (1 - t) * top + t * bot + (1 - s) * lef + s * rig - ((1 - s) * (1 - t) * px(x0 - 1, y0 - 1, c) + s * (1 - t) * px(x1 + 1, y0 - 1, c) + (1 - s) * t * px(x0 - 1, y1 + 1, c) + s * t * px(x1 + 1, y1 + 1, c))
        out[i + c] = Math.max(0, Math.min(255, Math.round(rgb[i + c] * (1 - m) + (P + nz) * m)))
      }
    }
  }
  return out
}
/* 얼굴 실제 크기 검사 (깊이를 안 뒤): 얼굴 폭 = 거리 r × 상자 폭 각도. 사람 얼굴(+ 여유)은 0.08~0.45 m — 벗어나면 무늬로 봐요.
   깊이가 없거나(하늘 · 무한) 이상하면 남겨요 (사생활 쪽으로 기울여요) */
export function faceKeep(boxes, r, W, H, { lo = 0.08, hi = 0.45 } = {}) {
  return boxes.filter((b) => { const cx = Math.round(((b.x + b.w / 2) % W + W) % W), cy = Math.max(0, Math.min(H - 1, Math.round(b.y + b.h / 2))), rr = r ? r[cy * W + cx] : NaN; if (!Number.isFinite(rr) || rr > 300) return true; const lat = (0.5 - (cy + 0.5) / H) * Math.PI, wm = rr * Math.cos(lat) * (b.w / W) * 2 * Math.PI; return wm >= lo && wm <= hi })
}
/* 흐릴 곳이 너무 많은 장면: 눈높이 띠(위도 ±35°) 안에서 흐릴 넓이 비율 · 개수 → 주 목록에서 빼요 (후보는 충분해요) */
export function privacyLoad(boxes, W, H, { band = 35 } = {}) {
  const y0 = H * (0.5 - band / 180), y1 = H * (0.5 + band / 180); let a = 0, n = 0
  for (const b of boxes) { const yy0 = Math.max(y0, b.y), yy1 = Math.min(y1, b.y + b.h); if (yy1 <= yy0) continue; a += b.w * (yy1 - yy0) * Math.PI / 4; n++ }
  return { frac: a / (W * (y1 - y0)), n }
}

/* 1단계 "모자이크 최소 충분 흐림" (사용자: 모자이크를 허용되는 최대한까지 완화): 얼굴마다 '얼굴 인식이 더는 같은 사람으로 못 알아보는' 가장 약한 흐림을 찾아요.
   embed(crop112) → 얼굴 특징 벡터 (SFace · OpenCV Zoo · Apache-2.0, 112×112 RGB 0–255). 원본과 흐린 얼굴의 코사인 유사도가 tau(기본 0.25 — SFace 같은 사람 기준 0.363 보다 한참 아래)
   밑이면 "충분". div 를 [lo, hi] 에서 이분 탐색 (div 가 클수록 σ = 폭/div 가 작아 약한 흐림) — 흐림이 셀수록 유사도가 줄어드는 단조성을 가정하고, 찾은 값은 한 번 더 확인해요.
   안전: 얼굴이 너무 작으면(폭 < 24px) 탐색 없이 기존 세기(lo), embed 가 없거나 실패하면 lo, 결과는 사람 확인 목록과 별개로 기록. 번호판은 이미 빈 판으로 지워서(plateBlank) 그대로. */
export function faceCrop112(rgb, W, H, b, { ch = 3, wrap = true } = {}) {
  const out = new Float32Array(3 * 112 * 112), X = (x) => (wrap ? ((x % W) + W) % W : Math.max(0, Math.min(W - 1, x)))
  for (let y = 0; y < 112; y++) for (let x = 0; x < 112; x++) { const sx = b.x + (x + 0.5) / 112 * b.w, sy = Math.max(0, Math.min(H - 1, b.y + (y + 0.5) / 112 * b.h)), i = (Math.floor(sy) * W + X(Math.floor(sx))) * ch
    for (let c = 0; c < 3; c++) out[c * 112 * 112 + y * 112 + x] = rgb[i + c] }
  return out
}
const cosv = (a, b) => { let ab = 0, aa = 0, bb = 0; for (let i = 0; i < a.length; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i] } return ab / Math.sqrt(aa * bb + 1e-12) }
export async function minimalDiv(rgb, W, H, b, embed, { lo = 8, hi = 32, tau = 0.25, iters = 5, wrap = true, minPx = 24 } = {}) {
  if (!embed || Math.min(b.w, b.h) < minPx) return { div: lo, sim: null, why: embed ? '작은 얼굴' : '인식 모델 없음' }
  try {
    const e0 = await embed(faceCrop112(rgb, W, H, b, { wrap })), simAt = async (div) => cosv(e0, await embed(faceCrop112(anonBlur(rgb, W, H, [{ ...b, div }], { wrap, noise: true }), W, H, b, { wrap })))
    const sh = await simAt(hi); if (sh < tau) return { div: hi, sim: +sh.toFixed(3), why: '가장 약한 흐림으로도 충분' }
    let a = lo, z = hi, sa = await simAt(lo); if (sa >= tau) return { div: lo, sim: +sa.toFixed(3), why: '기존 세기도 모자라 기존 세기 유지 · 사람 확인' }
    for (let k = 0; k < iters; k++) { const m = (a + z) / 2, sm = await simAt(m); if (sm < tau) { a = m; sa = sm } else z = m }
    return { div: +a.toFixed(2), sim: +sa.toFixed(3), why: '알아볼 수 없을 만큼만' }
  } catch (e) { return { div: lo, sim: null, why: '인식 실패 → 기존 세기' } }
}
