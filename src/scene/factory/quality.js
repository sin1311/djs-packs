// v1.0-D 품질 관문 (작은 사본 1024×512 로) — 해상도 · 2:1 · 좌우 이음새(진짜 360°) · 흐림(라플라시안 분산) · 노출 · 삼각대 구멍 ·
//  미적 점수(색감 Hasler–Süsstrunk · 대비 · 정보량) · 지각 해시(dHash 64비트, 중복) · 분위기 서술어용 색 통계
const lum = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v))
export function grayOf(rgb, W, H) { const g = new Float32Array(W * H); for (let i = 0; i < W * H; i++) g[i] = lum(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]); return g }
// dHash: 9×8 으로 줄여 가로 이웃 비교 → 64비트 ([hi, lo] 32비트 두 개)
export function dhash(rgb, W, H) {
  const g = grayOf(rgb, W, H), s = new Float32Array(72)
  for (let y = 0; y < 8; y++) for (let x = 0; x < 9; x++) { let a = 0, n = 0; for (let j = Math.floor(y * H / 8); j < Math.floor((y + 1) * H / 8); j += 2) for (let i = Math.floor(x * W / 9); i < Math.floor((x + 1) * W / 9); i += 2) { a += g[j * W + i]; n++ } s[y * 9 + x] = a / Math.max(1, n) }
  let hi = 0, lo = 0; for (let k = 0; k < 64; k++) { const y = k >> 3, x = k & 7, bit = s[y * 9 + x] > s[y * 9 + x + 1] ? 1 : 0; if (k < 32) hi = (hi | (bit << k)) >>> 0; else lo = (lo | (bit << (k - 32))) >>> 0 }
  return [hi, lo]
}
const pop = (v) => { v = v - ((v >>> 1) & 0x55555555); v = (v & 0x33333333) + ((v >>> 2) & 0x33333333); return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24 }
export const hamming = (a, b) => pop((a[0] ^ b[0]) >>> 0) + pop((a[1] ^ b[1]) >>> 0)
// 색 통계 → descriptors.statScores 의 입력
export function colorStats(rgb, W, H) {
  let L = 0, S = 0, T = 0, n = 0, spots = 0, snow = 0, green = 0, purple = 0, sky = 0, skyN = 0, botN = 0
  const g = grayOf(rgb, W, H)
  for (let y = 1; y < H - 1; y += 2) for (let x = 1; x < W - 1; x += 2) {
    const i = y * W + x, r = rgb[i * 3] / 255, gg = rgb[i * 3 + 1] / 255, b = rgb[i * 3 + 2] / 255, mx = Math.max(r, gg, b), mn = Math.min(r, gg, b), l = g[i] / 255, s = mx > 0 ? (mx - mn) / mx : 0
    L += l; S += s; T += (r - b) / (r + b + 0.05); n++
    let h = 0; if (mx > mn) { h = mx === r ? ((gg - b) / (mx - mn)) % 6 : mx === gg ? (b - r) / (mx - mn) + 2 : (r - gg) / (mx - mn) + 4; h = (h * 60 + 360) % 360 }
    const grad = Math.abs(g[i + 1] - g[i - 1]) + Math.abs(g[i + W] - g[i - W])
    if (y < H * 0.4) { skyN++; if (grad < 6 && l > 0.25) sky++ }
    if (l > 0.85 && grad > 40) spots++
    if (y > H * 0.55) { botN++; if (l > 0.7 && s < 0.12) snow++ }
    if (s > 0.2 && h > 70 && h < 160) green++
    if (s > 0.25 && h > 260 && h < 330) purple++
  }
  return { luma: L / n, sat: S / n, temp: clamp(T / n * 2.5, -1, 1), sky: sky / Math.max(1, skyN), spots: clamp(spots / n * 20), snow: snow / Math.max(1, botN), green: green / n, purple: clamp(purple / n * 3) }
}
// 관문: { pass, score 0–1, why[], stats, hash }. meta = { w, h }
export function gate(rgb, W, H, meta = {}, o = {}) {
  const minW = o.minW ?? 4000, g = grayOf(rgb, W, H), why = []
  const res = clamp((meta.w || W) / minW), aspect = meta.w && meta.h ? Math.abs(meta.w / meta.h - 2) < 0.04 : Math.abs(W / H - 2) < 0.04
  if (res < 1) why.push(`해상도 ${meta.w}px < ${minW}`); if (!aspect) why.push('2:1 이 아니에요 (등장방형 아님)')
  // 좌우 이음새: 첫 열과 끝 열의 차이 vs 보통 이웃 열 차이
  let seam = 0, nb = 0; for (let y = 0; y < H; y++) { seam += Math.abs(g[y * W] - g[y * W + W - 1]); nb += Math.abs(g[y * W + (W >> 1)] - g[y * W + (W >> 1) + 1]) + Math.abs(g[y * W + (W >> 2)] - g[y * W + (W >> 2) + 1]) } const seamR = seam / (nb / 2 + H * 0.5)
  if (seamR > 3) why.push(`좌우 이음새가 안 맞아요 (×${seamR.toFixed(1)}) — 360° 가 아니에요`)
  // 흐림: 라플라시안 분산 (가운데 띠, 극 근처는 늘어나서 제외)
  let ls = 0, ls2 = 0, ln = 0; for (let y = Math.floor(H * 0.3); y < H * 0.7; y++) for (let x = 1; x < W - 1; x++) { const i = y * W + x, v = g[i - 1] + g[i + 1] + g[i - W] + g[i + W] - 4 * g[i]; ls += v; ls2 += v * v; ln++ } const lapVar = ls2 / ln - (ls / ln) ** 2
  const sharp = clamp(Math.log10(lapVar + 1) / 2.4); if (lapVar < (o.blurMin ?? 50) && !o.sky) why.push(`흐려요 (라플라시안 분산 ${lapVar.toFixed(0)})`)
  // 노출 · 대비 · 정보량
  const hist = new Float64Array(256); let m = 0; for (let i = 0; i < W * H; i++) { const v = Math.min(255, Math.max(0, g[i] | 0)); hist[v]++; m += g[i] } m /= W * H
  let clip = 0, varL = 0, ent = 0; for (let v = 0; v < 256; v++) { const p = hist[v] / (W * H); if (v < 3 || v > 252) clip += p; varL += p * (v - m) ** 2; if (p > 0) ent -= p * Math.log2(p) }
  if (clip > 0.25) why.push(`노출이 날아갔어요 (${(clip * 100).toFixed(0)}%)`)
  // 삼각대 구멍: 바닥 5% 줄이 거의 검거나 거의 평평
  let dark = 0, bn = 0, bs = 0, bs2 = 0; for (let y = Math.floor(H * 0.95); y < H; y++) for (let x = 0; x < W; x += 2) { const v = g[y * W + x]; if (v < 12) dark++; bs += v; bs2 += v * v; bn++ } const bstd = Math.sqrt(Math.max(0, bs2 / bn - (bs / bn) ** 2)), hole = clamp(Math.max(dark / bn, bstd < 2.5 ? 1 : 0))
  // 색감 (Hasler–Süsstrunk)
  let rgM = 0, ybM = 0, rg2 = 0, yb2 = 0, k = 0; for (let i = 0; i < W * H; i += 3) { const r = rgb[i * 3], gg = rgb[i * 3 + 1], b = rgb[i * 3 + 2], rg = r - gg, yb = 0.5 * (r + gg) - b; rgM += rg; ybM += yb; rg2 += rg * rg; yb2 += yb * yb; k++ } rgM /= k; ybM /= k
  const colorful = Math.sqrt(rg2 / k - rgM * rgM + yb2 / k - ybM * ybM) + 0.3 * Math.sqrt(rgM * rgM + ybM * ybM)
  const aesth = 0.4 * clamp(colorful / 60) + 0.3 * clamp(Math.sqrt(varL) / 60) + 0.3 * clamp((ent - 5) / 2.5)
  const pass = !why.length, score = clamp(0.3 * sharp + 0.3 * aesth + 0.15 * res + 0.15 * (1 - clip * 2) + 0.1 * (1 - hole))
  return { pass, score: +score.toFixed(4), why, hole: +hole.toFixed(2), sharp: +sharp.toFixed(3), aesth: +aesth.toFixed(3), seam: +seamR.toFixed(2), stats: colorStats(rgb, W, H), hash: dhash(rgb, W, H) }
}
