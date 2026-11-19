// v1.0-D 분위기 서술어 다리 (설계 근거서 4-17) — 곡과 장면을 같은 "서술어 분포"로 옮겨 비교해요.
//  · 서술어는 6갈래(시간대 · 날씨 · 장소 · 감정 · 빛 · 색). 갈래마다 확률 분포(합 1).
//  · 장면 q(d): 장면 공장에서 CLIP 영-샷(갈래마다 softmax) · 모델이 없으면 색 통계 + 주제 기본값.
//  · 곡 p(d): 지금은 규칙(장르 · 분위기 · BPM · 에너지 · 섹션), v1.1 에서 CLAP 으로 바꿔 끼워요(같은 틀).
//  · 닮음 = Σ_g w_g · Σ_d √(p_gd · q_gd)  (갈래별 바타차리야 계수의 가중합, 0~1)
export const GROUPS = [
  ['time', '시간대', 0.18, [['dawn', '새벽', 'at dawn, soft blue light before sunrise'], ['morning', '아침', 'in the morning'], ['noon', '한낮', 'at noon under bright sun'], ['golden', '해 질 녘 금빛', 'during golden hour'], ['sunset', '노을', 'at sunset with an orange sky'], ['dusk', '땅거미', 'at dusk, blue hour'], ['night', '밤', 'at night'], ['midnight', '한밤', 'late at night, deep darkness']]],
  ['weather', '날씨', 0.12, [['clear', '맑음', 'with a clear sky'], ['cloudy', '구름', 'with scattered clouds'], ['overcast', '흐림', 'under an overcast grey sky'], ['fog', '안개', 'in thick fog'], ['rain', '비', 'in the rain with wet ground'], ['snow', '눈', 'with snow on the ground'], ['storm', '폭풍', 'under a dramatic stormy sky']]],
  ['place', '장소', 0.3, [['street', '도시 거리', 'a city street'], ['alley', '골목', 'a narrow alley'], ['downtown', '도심 고층', 'a downtown area with skyscrapers'], ['highway', '고속도로', 'a highway or wide road'], ['harbor', '항구', 'a harbor with boats'], ['beach', '해변', 'a beach by the sea'], ['lake', '호수 · 강', 'a lake or river shore'], ['forest', '숲', 'inside a forest'], ['mountain', '산', 'a mountain landscape'], ['field', '들판', 'an open field or meadow'], ['desert', '사막', 'a desert'], ['plaza', '광장', 'a public square'], ['market', '시장', 'a street market'], ['station', '역', 'a train station'], ['rooftop', '옥상', 'a rooftop overlooking a city'], ['park', '공원', 'a city park'], ['oldtown', '옛 거리', 'an old town with historic buildings'], ['temple', '사원 · 성당', 'a temple or church'], ['industrial', '공장 · 창고', 'an industrial site or warehouse'], ['parking', '주차장', 'a parking lot or garage'], ['tunnel', '터널 · 지하', 'a tunnel or underground passage'], ['club', '클럽', 'a nightclub with stage lights'], ['hall', '공연장', 'a concert hall or theater'], ['cafe', '카페 · 바', 'a cozy cafe or bar'], ['room', '실내 방', 'a living room interior'], ['stadium', '경기장', 'a stadium'], ['amusement', '놀이공원', 'an amusement park']]],
  ['mood', '감정', 0.24, [['lonely', '쓸쓸함', 'lonely and empty'], ['cozy', '아늑함', 'cozy and warm'], ['festive', '축제', 'festive and celebratory'], ['romantic', '낭만', 'romantic'], ['nostalgic', '향수', 'nostalgic and vintage'], ['mysterious', '신비', 'mysterious'], ['energetic', '활기', 'energetic and lively'], ['calm', '고요', 'calm and still'], ['melancholic', '우울', 'melancholic and sad'], ['dreamy', '몽환', 'dreamy and ethereal'], ['epic', '웅장', 'epic and grand'], ['gritty', '거침', 'gritty and raw'], ['futuristic', '미래', 'futuristic'], ['peaceful', '평화', 'peaceful'], ['tense', '긴장', 'tense and dark'], ['joyful', '기쁨', 'joyful and bright']]],
  ['light', '빛', 0.1, [['neon', '네온', 'lit by colorful neon signs'], ['lamps', '따뜻한 등', 'lit by warm street lamps'], ['strings', '전구 줄', 'with string lights'], ['candle', '촛불', 'by candlelight'], ['moon', '달빛', 'under moonlight'], ['sunrays', '햇살', 'with rays of sunlight'], ['citylights', '도시 불빛', 'with glittering city lights'], ['stage', '무대 조명', 'with stage spotlights and lasers']]],
  ['color', '색', 0.06, [['warm', '따뜻한 색', 'in warm orange tones'], ['cool', '차가운 색', 'in cool blue tones'], ['pastel', '파스텔', 'in soft pastel colors'], ['vivid', '선명', 'in vivid saturated colors'], ['mono', '무채색', 'in muted monochrome colors'], ['purple', '보라 · 분홍', 'in purple and pink tones']]],
]
export const DESC = GROUPS.flatMap(([g, , , list]) => list.map(([id, ko, prompt]) => ({ g, id, ko, prompt })))
export const IDX = Object.fromEntries(DESC.map((d, i) => [d.g + ':' + d.id, i]))
export const N = DESC.length
const RANGE = (() => { const r = {}; let k = 0; for (const [g, , , list] of GROUPS) { r[g] = [k, k + list.length]; k += list.length } return r })()
export const groupOf = (g) => RANGE[g]
export const prompt = (d) => `a 360 degree photo ${d.g === 'place' ? 'of ' + d.prompt : d.prompt}` /* CLIP 영-샷 문장 */

// 점수표(갈래:서술어 → 점수) → 갈래마다 softmax(점수/τ) → 길이 N 분포
export function dist(scores, tau = 0.6) {
  const out = new Float32Array(N)
  for (const [g, , , list] of GROUPS) { const [a, b] = RANGE[g]; let mx = -1e9; for (let i = a; i < b; i++) mx = Math.max(mx, scores[i] || 0); let s = 0; for (let i = a; i < b; i++) { out[i] = Math.exp(((scores[i] || 0) - mx) / tau); s += out[i] } for (let i = a; i < b; i++) out[i] /= s; void list }
  return out
}
const add = (sc, key, w) => { const i = IDX[key]; if (i != null) sc[i] = (sc[i] || 0) + w }
const addAll = (sc, obj, k = 1) => { for (const [key, w] of Object.entries(obj || {})) add(sc, key, w * k) }

// 장르 → 서술어 (gen/dna.js GNAME 의 장르 이름)
export const GENRE = {
  citypop: { 'place:street': 1.2, 'place:highway': 1, 'place:harbor': 0.8, 'place:rooftop': 0.7, 'time:sunset': 1, 'time:night': 0.8, 'time:dusk': 0.7, 'light:citylights': 1, 'light:neon': 0.5, 'mood:nostalgic': 1, 'mood:romantic': 0.7, 'color:purple': 0.6, 'color:warm': 0.5 },
  lofi: { 'place:cafe': 1, 'place:room': 1, 'place:alley': 0.8, 'place:street': 0.5, 'weather:rain': 1, 'weather:overcast': 0.5, 'time:night': 0.7, 'time:dusk': 0.5, 'light:lamps': 1, 'mood:cozy': 1, 'mood:calm': 0.8, 'mood:nostalgic': 0.6, 'color:pastel': 0.6, 'color:warm': 0.5 },
  house: { 'place:club': 1.3, 'place:rooftop': 0.8, 'place:beach': 0.5, 'time:night': 1, 'light:stage': 1, 'light:neon': 0.8, 'mood:energetic': 1, 'mood:festive': 0.8, 'color:vivid': 0.8, 'color:purple': 0.6 },
  techno: { 'place:industrial': 1.3, 'place:club': 1, 'place:tunnel': 1, 'place:parking': 0.6, 'time:midnight': 1, 'time:night': 0.7, 'light:stage': 0.8, 'mood:tense': 0.8, 'mood:gritty': 1, 'mood:futuristic': 0.8, 'color:mono': 0.8, 'color:cool': 0.6 },
  trap: { 'place:alley': 1, 'place:downtown': 0.9, 'place:parking': 1, 'place:street': 0.6, 'time:night': 1, 'time:midnight': 0.8, 'light:neon': 0.8, 'light:citylights': 0.6, 'mood:gritty': 1, 'mood:tense': 0.8, 'color:purple': 0.7, 'color:cool': 0.5 },
  ballad: { 'place:field': 1, 'place:lake': 1, 'place:beach': 0.7, 'place:park': 0.6, 'place:mountain': 0.6, 'time:sunset': 0.9, 'time:dusk': 0.7, 'time:dawn': 0.6, 'weather:rain': 0.5, 'weather:snow': 0.4, 'mood:melancholic': 1, 'mood:romantic': 0.8, 'mood:calm': 0.6, 'color:pastel': 0.6, 'color:warm': 0.4 },
  rock: { 'place:stadium': 1, 'place:highway': 1, 'place:desert': 0.8, 'place:industrial': 0.5, 'place:hall': 0.6, 'time:sunset': 0.6, 'time:noon': 0.5, 'weather:storm': 0.6, 'mood:energetic': 1, 'mood:epic': 0.9, 'mood:gritty': 0.6, 'light:stage': 0.6, 'color:vivid': 0.5 },
  kpop: { 'place:downtown': 1.1, 'place:amusement': 0.9, 'place:rooftop': 0.7, 'place:street': 0.6, 'time:noon': 0.7, 'time:night': 0.6, 'light:neon': 0.6, 'light:stage': 0.6, 'mood:joyful': 1, 'mood:energetic': 0.9, 'mood:futuristic': 0.4, 'color:vivid': 1, 'color:pastel': 0.6 },
  dnb: { 'place:tunnel': 1.1, 'place:highway': 1, 'place:downtown': 0.6, 'place:industrial': 0.6, 'time:night': 1, 'time:midnight': 0.6, 'light:citylights': 0.8, 'light:neon': 0.6, 'mood:energetic': 1, 'mood:futuristic': 1, 'mood:tense': 0.5, 'color:cool': 0.8 },
  reggaeton: { 'place:beach': 1.2, 'place:street': 0.6, 'place:rooftop': 0.6, 'place:club': 0.5, 'time:sunset': 0.9, 'time:noon': 0.6, 'weather:clear': 0.8, 'mood:festive': 1, 'mood:joyful': 0.8, 'mood:romantic': 0.5, 'light:sunrays': 0.6, 'color:warm': 0.8, 'color:vivid': 0.7 },
  waltz: { 'place:oldtown': 1.2, 'place:hall': 1, 'place:plaza': 0.8, 'place:temple': 0.6, 'place:park': 0.5, 'time:dusk': 0.6, 'time:night': 0.5, 'light:lamps': 0.8, 'light:candle': 0.6, 'mood:romantic': 1, 'mood:nostalgic': 0.8, 'mood:dreamy': 0.6, 'color:warm': 0.6 },
  carol: { 'place:market': 1.2, 'place:plaza': 1, 'place:oldtown': 0.8, 'place:temple': 0.5, 'weather:snow': 1.5, 'time:night': 1, 'time:dusk': 0.6, 'light:strings': 1.3, 'light:lamps': 0.8, 'light:candle': 0.5, 'mood:festive': 1, 'mood:cozy': 1, 'mood:joyful': 0.6, 'color:warm': 0.8 },
  boombap: { 'place:alley': 1, 'place:street': 0.9, 'place:parking': 0.6, 'place:station': 0.5, 'time:night': 0.6, 'time:dusk': 0.5, 'weather:overcast': 0.4, 'mood:gritty': 1, 'mood:nostalgic': 0.8, 'light:lamps': 0.5, 'color:mono': 0.6, 'color:warm': 0.4 },
  disco: { 'place:club': 1, 'place:rooftop': 0.8, 'place:hall': 0.5, 'place:downtown': 0.4, 'time:night': 1, 'light:stage': 0.9, 'light:neon': 0.7, 'mood:joyful': 1, 'mood:festive': 0.9, 'mood:nostalgic': 0.6, 'color:vivid': 0.9, 'color:purple': 0.8 },
}
const MOOD = {
  bright: { 'mood:joyful': 0.8, 'mood:peaceful': 0.4, 'time:noon': 0.5, 'time:golden': 0.6, 'weather:clear': 0.6, 'color:warm': 0.5, 'color:vivid': 0.4 },
  dark: { 'mood:mysterious': 0.7, 'mood:tense': 0.6, 'mood:melancholic': 0.5, 'time:night': 0.6, 'time:midnight': 0.6, 'weather:fog': 0.4, 'color:cool': 0.6, 'color:mono': 0.3 },
  calm: { 'mood:calm': 0.9, 'mood:peaceful': 0.7, 'mood:dreamy': 0.5, 'time:dawn': 0.5, 'time:dusk': 0.5, 'weather:fog': 0.3, 'color:pastel': 0.6 },
  energetic: { 'mood:energetic': 1, 'mood:festive': 0.5, 'mood:epic': 0.4, 'light:neon': 0.4, 'light:stage': 0.4, 'color:vivid': 0.6 },
}
const SECTION = { /* 섹션은 장르 기본값을 이길 만큼 세게: 같은 곡 안에서도 장면 분위기가 바뀌어야 해요 */
  intro: { 'mood:calm': 1, 'mood:mysterious': 0.8, 'time:dawn': 0.5, 'weather:fog': 0.5 },
  verse: {},
  pre: { 'mood:tense': 0.8, 'mood:dreamy': 0.4 },
  chorus: { 'mood:energetic': 1.2, 'mood:joyful': 0.9, 'mood:epic': 0.8, 'color:vivid': 0.6 },
  drop: { 'mood:energetic': 1.6, 'mood:epic': 1, 'light:stage': 0.7, 'light:neon': 0.5 },
  bridge: { 'mood:dreamy': 1.3, 'mood:melancholic': 0.8, 'mood:mysterious': 0.6, 'weather:rain': 0.4 },
  outro: { 'mood:nostalgic': 0.9, 'mood:calm': 1, 'time:dusk': 0.6 },
}
// 곡 특징 f = { genres: { 장르: 비중 }, mood, energy 0–1, bpm, section } → p(d)
export function songDist(f) {
  const sc = new Float32Array(N), gs = f.genres || {}, tot = Object.values(gs).reduce((a, b) => a + b, 0) || 1
  for (const [g, w] of Object.entries(gs)) addAll(sc, GENRE[g], (w / tot) * 1.6)
  addAll(sc, MOOD[f.mood], 1); addAll(sc, SECTION[f.section] || {}, 1)
  const e = f.energy ?? 0.6, b = f.bpm ?? 110
  add(sc, 'mood:energetic', (e - 0.5) * 1.2); add(sc, 'mood:calm', (0.5 - e) * 1.2); add(sc, 'mood:peaceful', (0.45 - e) * 0.8)
  if (b >= 125) { add(sc, 'light:neon', 0.4); add(sc, 'mood:energetic', 0.3) } if (b <= 85) { add(sc, 'time:night', 0.3); add(sc, 'weather:rain', 0.3); add(sc, 'mood:calm', 0.3) }
  return dist(sc, 0.55)
}
// 장면 주제 기본값 (장면 공장 주제 10개 · 합성 시험 장면)
export const THEME_Q = {
  city: { 'place:street': 1.4, 'place:highway': 0.8, 'time:sunset': 1.3, 'time:golden': 0.8, 'light:lamps': 0.6, 'mood:nostalgic': 0.8, 'mood:romantic': 0.6, 'color:warm': 1, 'color:purple': 0.4 },
  nightcity: { 'place:downtown': 1.2, 'place:street': 1, 'time:night': 1.4, 'light:citylights': 1.2, 'light:neon': 0.9, 'mood:energetic': 0.6, 'mood:mysterious': 0.5, 'color:purple': 0.8, 'color:cool': 0.6 },
  daycity: { 'place:downtown': 1.3, 'place:street': 0.9, 'time:noon': 1.4, 'weather:clear': 1, 'light:sunrays': 0.8, 'mood:joyful': 0.8, 'mood:energetic': 0.6, 'color:vivid': 0.8 },
  market: { 'place:market': 1.4, 'place:plaza': 1, 'weather:snow': 1.3, 'time:night': 1.1, 'light:strings': 1.2, 'mood:festive': 1, 'mood:cozy': 0.9, 'color:warm': 0.8 },
  club: { 'place:club': 1.5, 'place:hall': 0.8, 'time:midnight': 1, 'time:night': 0.8, 'light:stage': 1.4, 'light:neon': 0.9, 'mood:energetic': 1, 'mood:festive': 0.7, 'color:purple': 1, 'color:vivid': 0.6 },
  field: { 'place:field': 1.4, 'place:park': 0.5, 'time:sunset': 1.1, 'time:golden': 1, 'weather:clear': 0.6, 'mood:peaceful': 1, 'mood:romantic': 0.6, 'mood:melancholic': 0.4, 'color:warm': 0.8, 'color:pastel': 0.5 },
  forest: { 'place:forest': 1.5, 'place:mountain': 0.5, 'time:morning': 0.8, 'weather:fog': 0.6, 'light:sunrays': 0.8, 'mood:calm': 1, 'mood:mysterious': 0.6, 'mood:peaceful': 0.7, 'color:mono': 0.3 },
  coast: { 'place:beach': 1.4, 'place:harbor': 0.7, 'place:lake': 0.4, 'time:noon': 0.6, 'time:sunset': 0.8, 'weather:clear': 0.9, 'mood:joyful': 0.7, 'mood:peaceful': 0.6, 'mood:romantic': 0.5, 'color:vivid': 0.6, 'color:warm': 0.5 },
  industrial: { 'place:industrial': 1.5, 'place:parking': 0.8, 'place:tunnel': 0.5, 'time:night': 0.6, 'weather:overcast': 0.7, 'mood:gritty': 1.2, 'mood:tense': 0.7, 'mood:lonely': 0.6, 'color:mono': 1, 'color:cool': 0.5 },
  cozy: { 'place:room': 1.4, 'place:cafe': 1.1, 'time:night': 0.7, 'light:lamps': 1.2, 'light:candle': 0.6, 'mood:cozy': 1.4, 'mood:calm': 0.8, 'color:warm': 1 },
}
export const themeDist = (theme) => dist(Object.fromEntries(Object.entries(THEME_Q[theme] || {}).map(([k, w]) => [IDX[k], w])), 0.5)
// 색 통계(장면 공장이 재요) → 시간대 · 날씨 · 빛 · 색 쪽 점수 (모델이 없을 때 · 주제 기본값과 섞어요)
//  st = { luma 0–1, sat 0–1, temp −1(차가움)~1(따뜻함), sky 0–1(하늘 비율), spots 0–1(밝은 점 비율 = 조명), snow 0–1, green 0–1, purple 0–1 }
export function statScores(st) {
  const sc = new Float32Array(N), L = st.luma ?? 0.4
  add(sc, 'time:night', (0.25 - L) * 6 + (st.spots || 0) * 2); add(sc, 'time:midnight', (0.15 - L) * 6); add(sc, 'time:noon', (L - 0.5) * 5 + (st.sky || 0) * 0.5); add(sc, 'time:sunset', (st.temp || 0) * 2 * (L > 0.15 && L < 0.6 ? 1 : 0.3)); add(sc, 'time:golden', (st.temp || 0) * 1.5 * (L > 0.35 ? 1 : 0.2)); add(sc, 'time:dusk', (0.35 - Math.abs(L - 0.25)) * 2 - (st.temp || 0))
  add(sc, 'weather:overcast', (0.15 - (st.sat || 0)) * 5 * (st.sky || 0)); add(sc, 'weather:clear', ((st.sat || 0) - 0.2) * 3 * (st.sky || 0)); add(sc, 'weather:snow', (st.snow || 0) * 5)
  add(sc, 'light:neon', (st.purple || 0) * 3 + (st.spots || 0) * (st.sat || 0) * 4); add(sc, 'light:lamps', (st.spots || 0) * Math.max(0, st.temp || 0) * 4); add(sc, 'light:citylights', (st.spots || 0) * 3 * (L < 0.3 ? 1 : 0.3)); add(sc, 'light:sunrays', (L - 0.45) * 3)
  add(sc, 'color:warm', (st.temp || 0) * 3); add(sc, 'color:cool', -(st.temp || 0) * 3); add(sc, 'color:vivid', ((st.sat || 0) - 0.35) * 5); add(sc, 'color:mono', (0.12 - (st.sat || 0)) * 6); add(sc, 'color:purple', (st.purple || 0) * 4); add(sc, 'color:pastel', (L - 0.45) * 2 - Math.abs((st.sat || 0) - 0.25) * 4)
  add(sc, 'place:forest', (st.green || 0) * 3); add(sc, 'place:field', (st.green || 0) * 1.5 * (st.sky || 0) * 2)
  return sc
}
// 장면 분포: CLIP 점수(있으면) · 색 통계 · 주제 기본값을 섞어요
export function sceneDist({ clip, stats, theme } = {}) {
  const sc = new Float32Array(N)
  if (clip) for (let i = 0; i < N; i++) sc[i] += (clip[i] || 0) * 3
  if (stats) { const s = statScores(stats); for (let i = 0; i < N; i++) sc[i] += s[i] * (clip ? 0.3 : 1) }
  if (theme && THEME_Q[theme]) for (const [k, w] of Object.entries(THEME_Q[theme])) sc[IDX[k]] += w * (clip ? 0.3 : 1)
  return dist(sc, 0.5)
}
// 압축: 분포 → [[번호, 무게(0–255)], …] 상위 몇 개만 (목록 파일을 작게)
export function pack(q, keep = 4) { const out = []; for (const [g] of GROUPS) { const [a, b] = RANGE[g], idx = []; for (let i = a; i < b; i++) idx.push(i); idx.sort((x, y) => q[y] - q[x]); for (const i of idx.slice(0, keep)) if (q[i] > 0.02) out.push([i, Math.round(q[i] * 255)]) } return out }
export function unpack(arr) { const q = new Float32Array(N), has = new Uint8Array(N); for (const [i, w] of arr || []) { q[i] = w / 255; has[i] = 1 } for (const [g] of GROUPS) { const [a, b] = RANGE[g]; let s = 0, miss = 0; for (let i = a; i < b; i++) { s += q[i]; if (!has[i]) miss++ } const rest = Math.max(0.005 * (b - a), 1 - s); for (let i = a; i < b; i++) if (!has[i]) q[i] = miss ? rest / miss : 0; let t = 0; for (let i = a; i < b; i++) t += q[i]; for (let i = a; i < b; i++) q[i] /= t } return q } /* 남은 몫만 빠진 서술어에 나눠요 */
export function bridge(p, q) { let s = 0; for (const [g, , w] of GROUPS) { const [a, b] = RANGE[g]; let bc = 0; for (let i = a; i < b; i++) bc += Math.sqrt(p[i] * q[i]); s += w * bc } return s }
export function topDesc(q, k = 4) { return DESC.map((d, i) => [d, q[i], GROUPS.find((x) => x[0] === d.g)[2]]).sort((a, b) => b[1] * b[2] - a[1] * a[2]).slice(0, k).map(([d, w]) => ({ id: d.g + ':' + d.id, ko: d.ko, w: +w.toFixed(2) })) }
// v1.1 ② CLAP(오디오 ↔ 글) 용 서술어 문장: 장면 쪽 문장(사진 묘사)을 "음악" 묘사로 바꿔요. 글 임베딩은 Actions(clap-prep)에서 미리 계산해 앱에 실어요.
export function audioPrompt(d) {
  const p = d.prompt.replace(/^(a|an) /, '')
  if (d.g === 'mood') return `${d.prompt} music`
  if (d.g === 'place') return `music that fits ${d.prompt}`
  if (d.g === 'color') return `music that feels ${d.prompt.replace(/^in /, 'like ')}`
  return `music for a scene ${p}`
}
