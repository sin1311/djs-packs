// v1.0-C · D 장면 선택 — 점수 → softmax(점수/τ) 표집 × 최근 선택 감쇠 (1 − e^{−Δt/T}) − MMR(직전 장면과의 닮음 벌점)
//  · v1.0-D: 점수 = 서술어 다리(바타차리야, descriptors.js) 에 CSLS 허브 보정: 2·s(a, t) − r_T(a) − r_A(t).
//    r_T(곡) = 그 곡과 가장 닮은 장면 k 개의 평균, r_A(장면) = 그 장면과 가장 닮은 "원형 곡"(장르 × 분위기 × 섹션) k 개의 평균.
//  · v1.1 에서 곡 분포만 CLAP 으로 바꿔 끼워요 (같은 틀). 설계 근거서 4-5 · 4-17.
//  · Δt 는 "재생 횟수" 단위: 방금 고른 장면은 거의 0 → 같은 곡을 반복해도 장면이 돌아가요.
//  · 진단: kOccurrence(장면이 상위 k 에 든 횟수 분포)의 왜도 = 허브니스 지표.
import { songDist, themeDist, unpack, bridge, GENRE } from './descriptors.js'
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v))
export const SEL = { tau: 0.42, T: 8, lambda: 0.8, floor: 0.02, small: 40 } /* v1.0-D 시험으로 고른 값 (z 점수 · 장면 수에 맞춘 온도 · 감쇠): 가상 300곳 × 뽑기 2000번에서 장르에 맞는 주제 91% · 286곳 사용 / 3곳이면 같은 곡 20번에 3종 모두(300/300) · 연달아 같은 장면 평균 3.2번 */
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
// 장면끼리 닮음 (MMR 용): 같은 장면 · 주제 1, 아니면 서술어 분포의 바타차리야를 0~1 로 펴서
export function simQ(a, b) { if (!a || !b) return 0; if (a.id === b.id || (a.theme && a.theme === b.theme)) return 1; return clamp((bridge(qOf(a), qOf(b)) - 0.55) / 0.45) }
export function scoresOf(scenes, f, k = 10) { prepare(scenes); const p = f.p || songDist(f), raw = scenes.map((s) => bridge(p, qOf(s))), rT = topMean(raw, Math.min(k, scenes.length)), c = raw.map((x, i) => 2 * x - rT - scenes[i]._rA), m = c.reduce((a, b) => a + b, 0) / c.length, sd = Math.sqrt(c.reduce((a, b) => a + (b - m) ** 2, 0) / c.length) || 1; return c.map((x, i) => (x - m) / sd + ((f.boost && f.boost[scenes[i].theme]) || 0)) } /* boost: 추천 취향(좋아한 장면 주제) 가산점 */ /* z 점수: 장면이 3곳이든 3000곳이든 같은 온도가 통해요 */
// st = { n: 재생 횟수, last: { id: 마지막으로 고른 n }, prev: id } (저장해 두면 다음에도 이어져요)
export function probs(scenes, f, st = { n: 0, last: {}, prev: null }, o = SEL) {
  const prev = scenes.find((s) => s.id === st.prev), sc = scoresOf(scenes, f), tau = o.tau * Math.max(1, Math.sqrt((o.small || 40) / scenes.length)), z = scenes.map((s, i) => (sc[i] - o.lambda * (prev ? simQ(s, prev) : 0) /* 직전 장면 자신도 닮음 1 → 같은 장면 반복에 벌점 */) / tau) /* 장면이 적을수록 온도를 높여 z 간격을 좁혀요 (3곳 → ×2) */, mx = Math.max(...z)
  const T = o.T * Math.max(1, 24 / scenes.length), floor = o.floor * Math.min(1, scenes.length / 40) /* 장면이 적을수록 방금 본 장면을 더 오래 · 더 강하게 피해요 (3곳 → 감쇠 8배 · 하한 1/13) */
  const p = scenes.map((s, i) => { const dt = st.last[s.id] == null ? Infinity : st.n - st.last[s.id]; return Math.exp(z[i] - mx) * Math.max(floor, 1 - Math.exp(-dt / T)) })
  const sum = p.reduce((a, b) => a + b, 0) || 1; return p.map((x) => x / sum)
}
export function pick(scenes, f, st, rand = Math.random, o = SEL) {
  if (!scenes.length) return null
  const p = probs(scenes, f, st, o); let u = rand(), k = 0; for (; k < p.length - 1; k++) { u -= p[k]; if (u <= 0) break }
  const s = scenes[k]; st.last[s.id] = st.n; st.prev = s.id; st.n++; return { scene: s, p: p[k], all: p }
}
// 허브니스 진단: 여러 곡 특징 fs 에 대해 각 장면이 상위 k 에 든 횟수 → 왜도
export function kOccurrence(scenes, fs, k = 1) {
  const cnt = Object.fromEntries(scenes.map((s) => [s.id, 0]))
  for (const f of fs) { const sc = scoresOf(scenes, f); scenes.map((s, i) => [s.id, sc[i]]).sort((a, b) => b[1] - a[1]).slice(0, k).forEach(([id]) => cnt[id]++) }
  const v = Object.values(cnt), n = v.length, m = v.reduce((a, b) => a + b, 0) / n, sd = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / n) || 1
  return { cnt, skew: v.reduce((a, b) => a + ((b - m) / sd) ** 3, 0) / n }
}
export const featOf = (D) => ({ energy: clamp(D.energy ?? 0.6), bpm: D.bpm || 110, mood: D.mood || 'bright', genres: D.genres || {}, section: D.section || '', boost: D.planThemes ? Object.fromEntries(D.planThemes.map((t) => [t, 0.7])) : null })
