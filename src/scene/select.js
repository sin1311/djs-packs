// v1.0-C 장면 선택 — 곡 특징 규칙 점수 → softmax(점수/τ) 표집 × 최근 선택 감쇠 (1 − e^{−Δt/T}) − MMR(직전 장면과의 닮음 벌점)
//  · v1.1 에서 점수만 CLAP + CSLS 로 바꿔 끼워요 (같은 틀). 설계 근거서 4-5.
//  · Δt 는 "재생 횟수" 단위: 방금 고른 장면은 거의 0 → 같은 곡을 반복해도 장면이 돌아가요.
//  · 진단: kOccurrence(장면이 상위 k 에 든 횟수 분포)의 왜도 = 허브니스 지표.
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v))
export const SEL = { tau: 0.3, T: 8, lambda: 0.3, floor: 0.02 } /* 시험으로 고른 값: 첫 선택은 점수 1등이 73~87%, 같은 곡 20번에 3종 모두 (씨앗 300개 × 곡 3개) */
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
// st = { n: 재생 횟수, last: { id: 마지막으로 고른 n }, prev: id } (저장해 두면 다음에도 이어져요)
export function probs(scenes, f, st = { n: 0, last: {}, prev: null }, o = SEL) {
  const prev = scenes.find((s) => s.id === st.prev), z = scenes.map((s) => (score(s.mood, f) - o.lambda * (prev ? sim(s, prev) : 0) /* 직전 장면 자신도 닮음 1 → 같은 장면 반복에 벌점 */) / o.tau), mx = Math.max(...z)
  const p = scenes.map((s, i) => { const dt = st.last[s.id] == null ? Infinity : st.n - st.last[s.id]; return Math.exp(z[i] - mx) * Math.max(o.floor, 1 - Math.exp(-dt / o.T)) })
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
  for (const f of fs) scenes.map((s) => [s.id, score(s.mood, f)]).sort((a, b) => b[1] - a[1]).slice(0, k).forEach(([id]) => cnt[id]++)
  const v = Object.values(cnt), n = v.length, m = v.reduce((a, b) => a + b, 0) / n, sd = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / n) || 1
  return { cnt, skew: v.reduce((a, b) => a + ((b - m) / sd) ** 3, 0) / n }
}
export const featOf = (D) => ({ energy: clamp(D.energy ?? 0.6), bpm: D.bpm || 110, mood: D.mood || 'bright', genres: D.genres || {} })
