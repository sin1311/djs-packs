// v1.0-C LDI (층 깊이 영상): 깊이 경계로 2~3층 → 가려진 곳 채우기 → 층마다 색 · 거리 · 알파
//  · 층 나누기: log r 히스토그램의 다중 오츠(클래스 간 분산 최대) 문턱. 하늘은 늘 가장 먼 층.
//  · 층 k (0 = 가장 멂): 원본 화소 = 군집 k. 그보다 가까운 군집이 가린 곳은 채우기(밀고-당기기 피라미드)로 메워요.
//    알파: 층 0 은 어디나 1 (빈틈 없음). 층 k ≥ 1 은 자기 화소 + 가까운 층에 가려진 곳 중 band(°) 안쪽만 —
//    카메라가 ±0.3m 움직일 때 드러나는 폭(이동 × |1/Z₁ − 1/Z₂|, 몇 도)을 덮을 만큼.
//  · 채우기: 기본은 결정적인 밀고-당기기(push-pull). Actions 에서 LaMa(ONNX)가 있으면 색만 그것으로 바꿔요.

// 다중 오츠: 값 배열 → 문턱 (k 클래스 → k−1 개). 256 칸 히스토그램, 전수 탐색(k ≤ 3)
export function multiOtsu(vals, k = 3, bins = 256) {
  let lo = Infinity, hi = -Infinity; for (const v of vals) { if (v < lo) lo = v; if (v > hi) hi = v } if (!(hi > lo)) return []
  const hgm = new Float64Array(bins); for (const v of vals) hgm[Math.min(bins - 1, ((v - lo) / (hi - lo) * bins) | 0)]++
  const P = new Float64Array(bins + 1), S = new Float64Array(bins + 1); for (let i = 0; i < bins; i++) { P[i + 1] = P[i] + hgm[i]; S[i + 1] = S[i] + hgm[i] * i }
  const cls = (a, b) => { const p = P[b] - P[a]; return p > 0 ? (S[b] - S[a]) ** 2 / p : 0 }, at = (i) => lo + (i / bins) * (hi - lo)
  if (k === 2) { let best = -1, bt = 0; for (let t = 1; t < bins; t++) { const v = cls(0, t) + cls(t, bins); if (v > best) { best = v; bt = t } } return [at(bt)] }
  let best = -1, b1 = 0, b2 = 0; for (let t1 = 1; t1 < bins - 1; t1++) for (let t2 = t1 + 1; t2 < bins; t2++) { const v = cls(0, t1) + cls(t1, t2) + cls(t2, bins); if (v > best) { best = v; b1 = t1; b2 = t2 } }
  return [at(b1), at(b2)]
}
// 밀고-당기기 채우기: data(ch 채널 Float32) 의 mask=0 곳을 피라미드로 메워요. 가로는 감싸요(360°)
export function pushPull(data, mask, W, H, ch) {
  if (W <= 1 || H <= 1) { const out = data.slice(); let s = new Array(ch).fill(0), n = 0; for (let i = 0; i < W * H; i++) if (mask[i]) { for (let c = 0; c < ch; c++) s[c] += data[i * ch + c]; n++ } for (let i = 0; i < W * H; i++) if (!mask[i]) for (let c = 0; c < ch; c++) out[i * ch + c] = n ? s[c] / n : 0; return out }
  const w2 = Math.ceil(W / 2), h2 = Math.ceil(H / 2), d2 = new Float32Array(w2 * h2 * ch), m2 = new Float32Array(w2 * h2)
  for (let y = 0; y < h2; y++) for (let x = 0; x < w2; x++) { let ws = 0; const o = (y * w2 + x) * ch; for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) { const yy = Math.min(H - 1, 2 * y + dy), xx = (2 * x + dx) % W, i = yy * W + xx, m = mask[i]; if (m > 0) { ws += m; for (let c = 0; c < ch; c++) d2[o + c] += data[i * ch + c] * m } } if (ws > 0) { for (let c = 0; c < ch; c++) d2[o + c] /= ws; m2[y * w2 + x] = Math.min(1, ws) } }
  const f2 = pushPull(d2, m2, w2, h2, ch), out = data.slice()
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, m = Math.min(1, mask[i]); if (m >= 1) continue
    const fx = Math.max(0, (x - 0.5) / 2), fy = Math.max(0, Math.min(h2 - 1, (y - 0.5) / 2)), x0 = Math.floor(fx), y0 = Math.floor(fy), ax = fx - x0, ay = fy - y0, y1 = Math.min(h2 - 1, y0 + 1)
    for (let c = 0; c < ch; c++) { const p = (xx, yy) => f2[(yy * w2 + (xx % w2)) * ch + c], v = (p(x0, y0) * (1 - ax) + p(x0 + 1, y0) * ax) * (1 - ay) + (p(x0, y1) * (1 - ax) + p(x0 + 1, y1) * ax) * ay; out[i * ch + c] = data[i * ch + c] * m + v * (1 - m) }
  }
  return out
}
// 각거리(도) 변환: src(1) 화소에서의 근사 각거리 (2번 훑는 챔퍼, 가로 한 칸 = cos(위도)·Δλ)
export function angDist(src, W, H) {
  const D = new Float32Array(W * H).fill(1e9), dl = 360 / W, dv = 180 / H
  for (let i = 0; i < W * H; i++) if (src[i]) D[i] = 0
  const cw = (y) => Math.max(0.02, Math.cos((0.5 - (y + 0.5) / H) * Math.PI)) * dl
  for (let pass = 0; pass < 2; pass++) {
    for (let y = 0; y < H; y++) { const h = cw(y), dg = Math.hypot(h, dv); for (let x = 0; x < W; x++) { const i = y * W + x; let v = D[i]; v = Math.min(v, D[y * W + (x + W - 1) % W] + h); if (y > 0) { v = Math.min(v, D[i - W] + dv, D[(y - 1) * W + (x + W - 1) % W] + dg, D[(y - 1) * W + (x + 1) % W] + dg) } D[i] = v } }
    for (let y = H - 1; y >= 0; y--) { const h = cw(y), dg = Math.hypot(h, dv); for (let x = W - 1; x >= 0; x--) { const i = y * W + x; let v = D[i]; v = Math.min(v, D[y * W + (x + 1) % W] + h); if (y < H - 1) { v = Math.min(v, D[i + W] + dv, D[(y + 1) * W + (x + 1) % W] + dg, D[(y + 1) * W + (x + W - 1) % W] + dg) } D[i] = v } }
  }
  return D
}
// rgb(Uint8 ×3) · r(m, 하늘 ∞) → { layers: [{ rgba(Uint8 ×4), r(Float32) }], cuts(m) }
export function buildLDI(rgb, r, W, H, { layers = 3, band = 7, far = 400, minGap = 0.35 } = {}) {
  const N = W * H, lr = new Float32Array(N); for (let i = 0; i < N; i++) lr[i] = Math.log(Math.min(far, Number.isFinite(r[i]) ? r[i] : far))
  const vals = []; for (let i = 0; i < N; i += 3) if (Number.isFinite(r[i]) && r[i] < far) vals.push(lr[i])
  let cuts = vals.length > 100 ? multiOtsu(vals, layers) : []
  cuts = cuts.filter((c, i) => i === 0 || c - cuts[i - 1] > minGap) /* 거의 같은 문턱은 합쳐요 */
  const K = cuts.length + 1, cl = new Uint8Array(N) /* 0 = 가장 멂 */
  for (let i = 0; i < N; i++) { let k = 0; for (let q = cuts.length - 1; q >= 0; q--) if (lr[i] < cuts[q]) k = cuts.length - q; cl[i] = k }
  const out = []
  for (let k = 0; k < K; k++) {
    const own = new Float32Array(N); for (let i = 0; i < N; i++) own[i] = cl[i] === k ? 1 : 0
    const data = new Float32Array(N * 4); for (let i = 0; i < N; i++) { data[i * 4] = rgb[i * 3]; data[i * 4 + 1] = rgb[i * 3 + 1]; data[i * 4 + 2] = rgb[i * 3 + 2]; data[i * 4 + 3] = lr[i] }
    const fill = pushPull(data, own, W, H, 4), rgba = new Uint8Array(N * 4), rr = new Float32Array(N)
    const D = k > 0 ? angDist(own, W, H) : null
    for (let i = 0; i < N; i++) {
      for (let c = 0; c < 3; c++) rgba[i * 4 + c] = Math.max(0, Math.min(255, Math.round(fill[i * 4 + c])))
      rr[i] = Math.exp(fill[i * 4 + 3])
      let a = 1; if (k > 0) a = cl[i] === k ? 1 : cl[i] > k ? Math.max(0, 1 - D[i] / band) : 0
      rgba[i * 4 + 3] = Math.round(a * 255)
    }
    out.push({ rgba, r: rr })
  }
  return { layers: out, cuts: cuts.map(Math.exp), cl }
}
/* 장면마다 카메라가 움직여도 되는 거리 (시차 예산). 카메라가 t 만큼 움직이면 거리 Z₁ · Z₂ 인 이웃 두 점은 화면에서
   f · t · |1/Z₁ − 1/Z₂| 화소만큼 벌어져요. 같은 층 안에서 이어진 이웃(둘 다 불투명)의 이 값이 크면 그 사이 삼각형이 고무처럼 늘어나요.
   그 큰 쪽(99.5 백분위 D)을 ε 화소 안에 묶어 t = ε / (f · D), [lo, hi] 로 잘라요. f ≈ 1250 화소(1920 화면 · 시야 75°).
   메쉬 꼭짓점 간격쯤(4화소)마다 재요. */
export function parallaxBudget(layers, W, H, { f = 1250, eps = 6, lo = 0.08, hi = 0.3, step = 4 } = {}) {
  const ds = []
  for (const L of layers) for (let y = step; y < H - step; y += step) for (let x = 0; x < W; x += step) {
    const i = y * W + x, a = L.rgba[i * 4 + 3]; if (a < 200) continue
    for (const j of [y * W + ((x + step) % W), (y + step) * W + x]) { if (L.rgba[j * 4 + 3] < 200) continue; const r1 = L.r[i], r2 = L.r[j]; if (!(r1 > 0 && r2 > 0) || r1 > 300 || r2 > 300) continue; ds.push(Math.abs(1 / r1 - 1 / r2)) }
  }
  if (!ds.length) return { t: hi, D: 0 }
  ds.sort((p, q) => p - q); const D = ds[Math.floor(ds.length * 0.995)], t = D > 0 ? eps / (f * D) : hi
  return { t: +Math.max(lo, Math.min(hi, t)).toFixed(3), D: +D.toFixed(5) }
}
