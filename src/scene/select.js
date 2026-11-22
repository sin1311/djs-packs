// v1.0-C · D 장면 선택 — 점수 → softmax(점수/τ) 표집 × 최근 선택 감쇠 (1 − e^{−Δt/T}) − MMR(직전 장면과의 닮음 벌점)
//  · v1.0-D: 점수 = 서술어 다리(바타차리야, descriptors.js) 에 CSLS 허브 보정: 2·s(a, t) − r_T(a) − r_A(t).
//    r_T(곡) = 그 곡과 가장 닮은 장면 k 개의 평균, r_A(장면) = 그 장면과 가장 닮은 "원형 곡"(장르 × 분위기 × 섹션) k 개의 평균.
//  · v1.1 에서 곡 분포만 CLAP 으로 바꿔 끼워요 (같은 틀). 설계 근거서 4-5 · 4-17.
//  · Δt 는 "재생 횟수" 단위: 방금 고른 장면은 거의 0 → 같은 곡을 반복해도 장면이 돌아가요.
//  · 진단: kOccurrence(장면이 상위 k 에 든 횟수 분포)의 왜도 = 허브니스 지표.
import { songDist, themeDist, unpack, bridge, GENRE } from './descriptors.js'
import { mixDist, clapFor, CLAP_W } from '../ai/clap.js'
import { CLAP_HEARD } from '../ai/clapBias.js'
import { unpackFit } from './briefs.js'
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v))
export const SEL = { tau: 0.42, T: 8, lambda: 0.8, floor: 0.02, small: 40, H: 70, kappa: 0.7 } /* v1.0-D 시험으로 고른 값 (z 점수 · 장면 수에 맞춘 온도 · 감쇠): 가상 300곳 × 뽑기 2000번에서 장르에 맞는 주제 91% · 286곳 사용 / 3곳이면 같은 곡 20번에 3종 모두(300/300) · 연달아 같은 장면 평균 3.2번 */
// 곡 특징 f = { energy 0–1, bpm, mood, genres: { 장르: 비중 } } × 장면 분위기 m = { energy, bpm: [lo, hi], moods: [], genres: [] }
export function score(m, f) {
  if (!m) return 0
  const e = 1 - Math.abs((f.energy ?? 0.6) - (m.energy ?? 0.6)) * 1.6
  const [lo, hi] = m.bpm || [60, 200], b = f.bpm ?? 110, bp = b >= lo && b <= hi ? 1 : Math.exp(-Math.min(Math.abs(b - lo), Math.abs(b - hi)) / 18)
  const md = (m.moods || []).includes(f.mood) ? 1 : 0
  let gs = 0, gt = 0; for (const [g, w] of Object.entries(f.genres || {})) { gt += w; if ((m.genres || []).includes(g)) gs += w }
  return 0.3 * e + 0.2 * bp + 0.2 * md + 0.3 * (gt > 0 ? gs / gt : 0)
}
// 장면끼리 닮음 (MMR 용): 주제 같으면 1, 아니면 장르 · 분위기 자카드
export function sim(a, b) {
  if (!a || !b) return 0; if (a.id === b.id || (a.theme && a.theme === b.theme)) return 1
  const A = new Set([...(a.mood?.genres || []), ...(a.mood?.moods || [])]), B = new Set([...(b.mood?.genres || []), ...(b.mood?.moods || [])]); let i = 0; for (const x of A) if (B.has(x)) i++
  return A.size + B.size ? i / (A.size + B.size - i) : 0
}
// 장면에 분포 q 와 허브 정도 r_A 를 붙여요 (목록을 받을 때 한 번)
const PROTO = (() => { const out = []; for (const g of Object.keys(GENRE)) for (const mood of ['bright', 'dark', 'calm', 'energetic']) for (const section of ['verse', 'chorus', 'bridge']) out.push(songDist({ genres: { [g]: 1 }, mood, section, energy: mood === 'energetic' ? 0.85 : mood === 'calm' ? 0.35 : 0.6, bpm: mood === 'energetic' ? 128 : mood === 'calm' ? 80 : 108 })); return out })()
const topMean = (arr, k) => { const a = Float64Array.from(arr).sort(); let s = 0; const n = Math.min(k, a.length); for (let i = a.length - n; i < a.length; i++) s += a[i]; return n ? s / n : 0 }
export const qOf = (s) => s._q || (s._q = s.desc ? unpack(s.desc) : themeDist(s.theme || (s.mood && s.mood.theme) || 'city'))
export function prepare(scenes, k = 10) { for (const s of scenes) { const q = qOf(s); if (s._rA == null) s._rA = topMean(PROTO.map((p) => bridge(p, q)), k) } return scenes }
// 곡 듣기(CLAP)를 섞은 p 는 규칙 원형과 모양이 달라 허브 장면도 달라요 → 들은 곡 원형(clapBias.js 의 72개)으로 r_A 를 따로, 원형 전체 평균으로
//  (상위 10개 평균이면 왜도 5.2, 전체 평균이면 4.25 — 규칙만 4.17 과 같은 수준. 설계 근거서 4-21)
const SEC_E0 = { intro: 0.35, verse: 0.55, pre: 0.65, chorus: 0.85, drop: 0.95, bridge: 0.5, outro: 0.3 }
let HEARD = null
const heardProtos = () => HEARD || (HEARD = CLAP_HEARD.map(([g, mood, section, q]) => { const c = Float32Array.from(q, (x) => Math.max(1e-4, x / 1000)); return mixDist(songDist({ genres: { [g]: 1 }, mood, section, energy: SEC_E0[section] ?? 0.6, bpm: GENRE_BPM[g] || 110 }), c) }))
const GENRE_BPM = { trap: 140, lofi: 80, house: 124, techno: 132, citypop: 110, ballad: 72, rock: 140, kpop: 114, dnb: 172, reggaeton: 95, carol: 104, waltz: 120 }
const rOf = (s, f) => { if (!f.heard) return s._rA; if (s._rC == null) { const H = heardProtos(), q = qOf(s); s._rC = topMean(H.map((p) => bridge(p, q)), H.length) } return s._rC }
// 장면끼리 닮음 (MMR 용): 같은 장면 · 주제 1, 아니면 서술어 분포의 바타차리야를 0~1 로 펴서
export function simQ(a, b) { if (!a || !b) return 0; if (a.id === b.id || (a.theme && a.theme === b.theme)) return 1; return clamp((bridge(qOf(a), qOf(b)) - 0.55) / 0.45) }
/* v2.0 큐레이션(briefs.js): 장면 공장이 CLIP 으로 매긴 장르 적합도 g(상위 3개) · 감성 ap 를 점수에 더해요.
   곡의 장르 비중과 장면 적합도의 내적 → 장면들 사이 z 점수 × FIT.w, 감성은 0.7 기준 × FIT.ap. g 가 없는 장면(합성 · 옛 목록)은 0. */
export const FIT = { w: 1.4, ap: 0.8, ae: 0.6, min: 6, top: 0.18 } /* ae: 미학 점수(6.0~8.75)를 z 처럼 — 같은 장르 안에서 7점대 사진이 먼저 · 더 자주 (2026-11-22) */
export function fitOf(s, genres) { if (!s || !s.g) return 0; const F = s._fit || (s._fit = unpackFit(s.g)); let a = 0, t = 0; for (const [g, w] of Object.entries(genres || {})) { t += w; a += w * (F[g] || 0) } return t > 0 ? a / t : 0 }
const zs = (a) => { const m = a.reduce((x, y) => x + y, 0) / a.length, sd = Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length); return sd > 1e-6 ? a.map((x) => (x - m) / sd) : a.map(() => 0) }
/* 장르에 맞는 장면만 남기기: 곡 장르 적합도가 top 이상인 큐레이션 장면이 min 곳 넘게 있으면 그것들만 (없으면 전체 그대로) */
export function genrePool(scenes, f, o = FIT) { if (!f || !f.genres || !Object.keys(f.genres).length) return scenes; const ok = scenes.filter((s) => s.g && fitOf(s, f.genres) >= o.top); return ok.length >= o.min ? ok : scenes }
export function scoresOf(scenes, f, k = 10) { prepare(scenes); const p = f.p || songDist(f), raw = scenes.map((s) => bridge(p, qOf(s))), rT = topMean(raw, Math.min(k, scenes.length)), c = raw.map((x, i) => 2 * x - rT - rOf(scenes[i], f)), m = c.reduce((a, b) => a + b, 0) / c.length, sd = Math.sqrt(c.reduce((a, b) => a + (b - m) ** 2, 0) / c.length) || 1
  const cur = scenes.some((s) => s.g), gz = cur ? zs(scenes.map((s) => fitOf(s, f.genres))) : null
  return c.map((x, i) => (x - m) / sd + ((f.boost && f.boost[scenes[i].theme]) || 0) + (cur ? FIT.w * gz[i] + (scenes[i].ap != null ? FIT.ap * (scenes[i].ap / 255 - 0.7) / 0.15 : 0) + (scenes[i].ae != null ? FIT.ae * (scenes[i].ae / 25 - 6.8) / 0.55 : 0) : 0)) } /* boost: 추천 취향(좋아한 장면 주제) 가산점 */ /* z 점수: 장면이 3곳이든 3000곳이든 같은 온도가 통해요 */
// st = { n: 재생 횟수, last: { id: 마지막으로 고른 n }, prev: id } (저장해 두면 다음에도 이어져요)
// v1.1 ② 온라인 수송(사용량 퍼텐셜): 장면마다 할인된 사용 횟수 u_s = Σ exp(−(n − t_i)/H) 를 z 에서 κ·u_s 만큼 빼요.
//  고정 퍼텐셜(불균형 Sinkhorn)과 같은 최적 수송을 "이 사람이 실제로 트는 곡"으로 온라인으로 푸는 확률적 미러 하강 — 클럽 곡만 이어 트는 날처럼
//  드문 장면에 수요가 몰릴 때만 퍼뜨리고, 섞어 트는 날엔 거의 0. 짧은 감쇠(T, 곱셈)는 "방금 본 것 피하기", 이것은 "오래 보면 고르게".
//  장면이 적은 팩(200곳 미만)은 κ 를 비례해 줄여요(3곳 시험 팩은 예전 그대로). 설계 근거서 4-22.
const useOf = (st, id, o) => { const e = st.u && st.u[id]; return e ? e[0] * Math.exp(-(st.n - e[1]) / o.H) : 0 }
export function probs(scenes, f, st = { n: 0, last: {}, prev: null }, o = SEL) {
  const prev = scenes.find((s) => s.id === st.prev), sc = scoresOf(scenes, f), tau = o.tau * Math.max(1, Math.sqrt((o.small || 40) / scenes.length)), kap = (o.kappa || 0) * Math.min(1, scenes.length / 200), z = scenes.map((s, i) => (sc[i] - kap * useOf(st, s.id, o) - o.lambda * (prev ? simQ(s, prev) : 0) /* 직전 장면 자신도 닮음 1 → 같은 장면 반복에 벌점 */) / tau) /* 장면이 적을수록 온도를 높여 z 간격을 좁혀요 (3곳 → ×2) */, mx = Math.max(...z)
  const T = o.T * Math.max(1, 24 / scenes.length), floor = o.floor * Math.min(1, scenes.length / 40) /* 장면이 적을수록 방금 본 장면을 더 오래 · 더 강하게 피해요 (3곳 → 감쇠 8배 · 하한 1/13) */
  const p = scenes.map((s, i) => { const dt = st.last[s.id] == null ? Infinity : st.n - st.last[s.id]; return Math.exp(z[i] - mx) * Math.max(floor, 1 - Math.exp(-dt / T)) })
  const sum = p.reduce((a, b) => a + b, 0) || 1; return p.map((x) => x / sum)
}
export function pick(scenes, f, st, rand = Math.random, o = SEL) {
  if (!scenes.length) return null
  const p = probs(scenes, f, st, o); let u = rand(), k = 0; for (; k < p.length - 1; k++) { u -= p[k]; if (u <= 0) break }
  const s = scenes[k]; st.last[s.id] = st.n; st.prev = s.id
  if (o.kappa) { const u = (st.u = st.u || {}); u[s.id] = [useOf(st, s.id, o) + 1, st.n]; if (st.n % 50 === 0) for (const id in u) if (useOf(st, id, o) < 0.02) delete u[id] } /* 사라질 만큼 작아진 것은 지워 저장소를 작게 */
  st.n++; return { scene: s, p: p[k], all: p }
}
// 허브니스 진단: 여러 곡 특징 fs 에 대해 각 장면이 상위 k 에 든 횟수 → 왜도
export function kOccurrence(scenes, fs, k = 1) {
  const cnt = Object.fromEntries(scenes.map((s) => [s.id, 0]))
  for (const f of fs) { const sc = scoresOf(scenes, f); scenes.map((s, i) => [s.id, sc[i]]).sort((a, b) => b[1] - a[1]).slice(0, k).forEach(([id]) => cnt[id]++) }
  const v = Object.values(cnt), n = v.length, m = v.reduce((a, b) => a + b, 0) / n, sd = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / n) || 1
  return { cnt, skew: v.reduce((a, b) => a + ((b - m) / sd) ** 3, 0) / n }
}
export function featOf(D) { /* D.clap: 곡 듣기(CLAP) 결과가 있으면 규칙 p 와 기하 평균으로 섞은 p 를 함께 실어요 (scoresOf · 변주가 f.p 를 먼저 써요) */
  const f = { energy: clamp(D.energy ?? 0.6), bpm: D.bpm || 110, mood: D.mood || 'bright', genres: D.genres || {}, section: D.section || '', boost: D.planThemes ? Object.fromEntries(D.planThemes.map((t) => [t, 0.7])) : null }
  const c = D.clap && clapFor(D.clap, f.section); if (c) { f.p = mixDist(songDist(f), c, D.clapW ?? CLAP_W); f.heard = true }
  return f
}
