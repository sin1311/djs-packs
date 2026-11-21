/* 작품성 검사 (2026-11-21 사용자: "피사체를 너무 가까이서 찍은 건 빼 줘 · 사람들이 통상적으로 잘 찍었다 · 작품으로 남겨도 될 만하다고 하는 사진의
   공통점을 확실하게 충족하는 사진만")
   사진 평론 · 사진 대회 심사 기준에서 공통으로 꼽는 것을 잴 수 있는 것으로 옮겼어요:
   ① 초점 — 어딘가는 또렷해요(배경을 일부러 흐린 사진도 주제는 또렷)            → techStats.sharp (4×4 칸 중 가장 또렷한 칸의 라플라시안 분산)
   ② 노출 — 하이라이트가 하얗게 날아가거나 그늘이 새까맣게 뭉개지지 않아요       → clipHi · clipLo (밤 사진은 어두운 쪽을 넉넉히)
   ③ 대비 · 색 — 밋밋하지 않고(대비) · 바래지도 과하지도 않은 색(채도 · 컬러풀니스) → contrast · sat · oversat · colorful (Hasler–Süsstrunk 2003)
   ④ 구도 · 빛 · 거리감 — 넓게 담아 깊이가 있고(가까이 찍은 클로즈업 아님) · 사람들이 잘 찍었다고 느끼는 사진
      → CLIP 비교: 전문 사진(구도 · 빛 · 이끄는 선 · 골든아워) vs 아마추어 스냅(흔들림 · 기울어짐 · 어수선함) = pro,
                   넓은 장면(풍경 · 도시 풍경 · 거리 전경) vs 가까이(클로즈업 · 접사 · 음식 · 간판 · 얼굴) = close
   ⑤ 미학 점수 — 사람들의 사진 평점으로 학습한 LAION Aesthetics V1(선형 · CLIP ViT-B/32 · MIT) = aes (1~10)
   순수 함수 (DOM · 모델 없음). 시험: unit39. */
import AES from './aes-b32.json' with { type: 'json' }

/* ④ CLIP 비교 문장 (영어) — 갈래마다 자기들끼리만 softmax 해서 다른 판정(감성 · 거절 · 장르)의 확률을 흔들지 않아요 */
export const PRO = ['an award winning professional photograph', 'a stunning photograph with beautiful composition and lighting', 'a professional travel photography shot', 'a cinematic wide shot with depth and leading lines', 'a beautiful photo taken at golden hour or blue hour', 'a well composed photo following the rule of thirds']
export const AMATEUR = ['an amateur snapshot taken with a phone', 'a poorly composed casual photo', 'a blurry out of focus photo', 'an overexposed washed out photo', 'a dark underexposed noisy photo', 'a tilted crooked photo', 'a cluttered messy photo', 'a low quality compressed image']
export const WIDE = ['a wide shot of a street', 'a wide view of a place', 'a cityscape', 'a landscape view', 'an establishing shot of a scene', 'a view of a square with buildings']
export const CLOSE = ['a close-up photo of an object', 'a macro photo', 'a close-up of food on a plate', 'a close-up of a sign or text', 'a close-up portrait of a person', 'a detail shot filling the frame', 'a close-up of a flower or plant', 'a close-up of a product', 'a close-up of a market stall with food', 'a close-up of a few people']
export const CRAFT_PROMPTS = () => [...PRO.map((p) => ['pro', p]), ...AMATEUR.map((p) => ['am', p]), ...WIDE.map((p) => ['wide', p]), ...CLOSE.map((p) => ['close', p])]
const CRAFT_KEYS = new Set(['pro', 'am', 'wide', 'close'])
export const isCraftKey = (k) => CRAFT_KEYS.has(k)

/* 두 갈래 비: a 갈래 확률 / (a + b), 두 갈래 문장만으로 softmax(×100) */
export function axis(lg, keys, a, b) { let mx = -Infinity; for (let i = 0; i < lg.length; i++) if (keys[i] === a || keys[i] === b) mx = Math.max(mx, lg[i]); let pa = 0, pb = 0; for (let i = 0; i < lg.length; i++) { if (keys[i] === a) pa += Math.exp(lg[i] - mx); else if (keys[i] === b) pb += Math.exp(lg[i] - mx) } return pa + pb > 0 ? pa / (pa + pb) : 0.5 }

/* ⑤ 미학 점수: 정규화한 CLIP ViT-B/32 이미지 임베딩 · w + b (LAION-AI/aesthetic-predictor sa_0_4_vit_b_32_linear) */
export function aesScore(v) { if (!v || v.length !== AES.w.length) return null; let s = AES.b; for (let k = 0; k < v.length; k++) s += AES.w[k] * v[k]; return +s.toFixed(3) }

/* ①~③ 화소 통계: rgb(Uint8, 3채널) W×H — 긴 변 512 정도로 줄인 것을 넣어요 */
export function techStats(rgb, W, H) {
  const n = W * H, L = new Float32Array(n); let sumL = 0, sumL2 = 0, hi = 0, lo = 0, sat = 0, over = 0, rgS = 0, rgS2 = 0, ybS = 0, ybS2 = 0
  for (let i = 0; i < n; i++) {
    const r = rgb[i * 3] / 255, g = rgb[i * 3 + 1] / 255, b = rgb[i * 3 + 2] / 255, l = 0.2126 * r + 0.7152 * g + 0.0722 * b; L[i] = l; sumL += l; sumL2 += l * l
    if (l > 0.985) hi++; if (l < 0.02) lo++
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), s = mx > 0 ? (mx - mn) / mx : 0; sat += s; if (s > 0.95 && mx > 0.5) over++
    const rg = (r - g) * 255, yb = (0.5 * (r + g) - b) * 255; rgS += rg; rgS2 += rg * rg; ybS += yb; ybS2 += yb * yb
  }
  const meanL = sumL / n, contrast = Math.sqrt(Math.max(0, sumL2 / n - meanL * meanL))
  const sdRg = Math.sqrt(Math.max(0, rgS2 / n - (rgS / n) ** 2)), sdYb = Math.sqrt(Math.max(0, ybS2 / n - (ybS / n) ** 2)), colorful = Math.sqrt(sdRg ** 2 + sdYb ** 2) + 0.3 * Math.sqrt((rgS / n) ** 2 + (ybS / n) ** 2)
  /* 초점: 4×4 칸마다 라플라시안 분산 → 가장 또렷한 칸(주제) · 가운데값(전체) */
  const tiles = []; const tw = Math.floor(W / 4), th = Math.floor(H / 4)
  for (let ty = 0; ty < 4; ty++) for (let tx = 0; tx < 4; tx++) { let s = 0, s2 = 0, c = 0; for (let y = Math.max(1, ty * th); y < Math.min(H - 1, (ty + 1) * th); y++) for (let x = Math.max(1, tx * tw); x < Math.min(W - 1, (tx + 1) * tw); x++) { const i = y * W + x, lap = (4 * L[i] - L[i - 1] - L[i + 1] - L[i - W] - L[i + W]) * 255; s += lap; s2 += lap * lap; c++ } tiles.push(c ? s2 / c - (s / c) ** 2 : 0) }
  tiles.sort((a, b) => a - b)
  return { sharp: +tiles[15].toFixed(1), sharpMed: +tiles[8].toFixed(1), clipHi: +(hi / n).toFixed(4), clipLo: +(lo / n).toFixed(4), meanL: +meanL.toFixed(3), contrast: +contrast.toFixed(3), sat: +(sat / n).toFixed(3), oversat: +(over / n).toFixed(4), colorful: +colorful.toFixed(1) }
}
/* 기술 판정 (검토 시트로 정한 값 — Actions review 가지): 밤 사진(평균 밝기 < 0.22)은 어두운 쪽 뭉개짐을 넉넉히 */
export const TECH_CUT = { sharp: 60, clipHi: 0.06, clipLo: 0.12, clipLoNight: 0.45, contrast: 0.11, satMin: 0.08, satMax: 0.72, oversat: 0.08, colorful: 18 }
export function techVerdict(t, cut = TECH_CUT) {
  if (!t) return { ok: true, why: '' }
  const night = t.meanL < 0.22
  if (t.sharp < cut.sharp) return { ok: false, why: '초점: 또렷한 곳이 없어요' }
  if (t.clipHi > cut.clipHi) return { ok: false, why: '노출: 하이라이트가 날아갔어요' }
  if (t.clipLo > (night ? cut.clipLoNight : cut.clipLo)) return { ok: false, why: '노출: 그늘이 뭉개졌어요' }
  if (t.contrast < cut.contrast) return { ok: false, why: '대비: 밋밋해요' }
  if (t.sat < cut.satMin && t.colorful < cut.colorful) return { ok: false, why: '색: 바랬어요' }
  if (t.sat > cut.satMax || t.oversat > cut.oversat) return { ok: false, why: '색: 너무 과해요' }
  return { ok: true, why: '' }
}
