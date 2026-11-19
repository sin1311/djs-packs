// v1.0-D 선별: 중복 빼기 → 분위기 칸 채우기(탐욕 커버리지, 부분 모듈러) → 남은 자리는 품질 순 → 빈 칸 보고서(다음 수집 검색어)
//  · 칸 = 서술어 하나. 칸마다 목표 장면 수 Tᵢ (갈래 무게에 비례). 장면 c 의 서술어 분포 q_c 로 "기대 개수" cov_i = Σ q_ci.
//  · 이득(c) = Σ_i min(q_ci, max(0, Tᵢ − cov_i)) + β · 품질 — 부분 모듈러라 탐욕이 최적의 (1 − 1/e) 이상을 보장해요.
import { hamming } from './quality.js'
import { DESC, GROUPS, N } from '../descriptors.js'

export function dedupe(cands, maxHam = 6) {
  const keep = [], byQ = [...cands].sort((a, b) => b.q - a.q)
  outer: for (const c of byQ) { for (const k of keep) if (c.hash && k.hash && hamming(c.hash, k.hash) <= maxHam) { k.dups = (k.dups || 0) + 1; continue outer } keep.push(c) }
  return keep
}
/* 칸 목표: 갈래 하나의 서술어 확률은 장면마다 합 1 이라, 장면 S 곳이면 갈래 전체가 S 만큼 채워져요.
   목표 = share · S / (그 갈래 서술어 수) · (0.6 + 0.4 · 갈래 무게 / 최대) — 고르게 나눈 몫의 절반쯤을 바닥으로 둬요.
   (예전엔 칸마다 6곳 고정이라 장면 1488곳이면 모든 칸이 "다 참" → 빈 칸 보고서가 비었어요) */
export function targets(size = 60, share = 0.5) { const T = new Float32Array(N); const wmax = Math.max(...GROUPS.map((g) => g[2])), k = {}; for (const d of DESC) k[d.g] = (k[d.g] || 0) + 1; DESC.forEach((d, i) => { const w = GROUPS.find((g) => g[0] === d.g)[2]; T[i] = share * size / k[d.g] * (0.6 + 0.4 * w / wmax) }); return T }
// cands: [{ id, q(품질 0–1), d(Float32Array N 분포), … }] → { picked, cov, gaps }
export function curate(cands, { total = 2500, size = 0, share = 0.5, beta = 0.6, init = null } = {}) { /* size: 다 모였을 때 장면 수 (이미 있는 것 + 이번 total) */
  const T = targets(size || total, share), cov = init ? Float32Array.from(init) : new Float32Array(N) /* init: 이미 있는 장면들이 채운 몫 (이어서 모으기) */, picked = [], left = new Set(cands.map((_, i) => i))
  const gain = (c) => { let s = 0; for (let i = 0; i < N; i++) { const need = T[i] - cov[i]; if (need > 0) s += Math.min(c.d[i], need) } return s + beta * c.q }
  // 지연 탐욕: 이득은 줄어들기만 해서(부분 모듈러), 큐 맨 위만 다시 계산하면 돼요
  const heap = cands.map((c, i) => ({ i, g: gain(c) })).sort((a, b) => b.g - a.g)
  while (picked.length < total && heap.length) {
    const top = heap[0], g = gain(cands[top.i])
    if (heap.length > 1 && g < heap[1].g - 1e-9) { top.g = g; heap.sort((a, b) => b.g - a.g); continue }
    heap.shift(); if (!left.has(top.i)) continue; left.delete(top.i); const c = cands[top.i]; picked.push(c); for (let i = 0; i < N; i++) cov[i] += c.d[i]
  }
  const gaps = DESC.map((d, i) => ({ key: d.g + ':' + d.id, ko: d.ko, have: +cov[i].toFixed(1), want: +T[i].toFixed(1), term: d.prompt })).filter((x) => x.have < x.want).sort((a, b) => a.have / a.want - b.have / b.want)
  return { picked, cov, gaps }
}
// 빈 칸 → 다음 수집 검색어 (위키미디어 검색어로 바로 써요). 감정 · 색 칸은 검색어로 찾기 어려워서 장소 · 시간대 · 날씨 · 빛 칸만
const TERM = { street: 'city street', alley: 'alley', downtown: 'skyline downtown', highway: 'highway road', harbor: 'harbor port', beach: 'beach', lake: 'lake river', forest: 'forest', mountain: 'mountain', field: 'meadow field', desert: 'desert', plaza: 'square plaza', market: 'market', station: 'railway station', rooftop: 'rooftop', park: 'park', oldtown: 'old town', temple: 'church interior', industrial: 'factory industrial', parking: 'parking garage', tunnel: 'tunnel', club: 'nightclub', hall: 'concert hall', cafe: 'cafe interior', room: 'living room', stadium: 'stadium', amusement: 'amusement park',
  dawn: 'dawn', morning: 'morning', noon: 'midday', golden: 'golden hour', sunset: 'sunset', dusk: 'blue hour', night: 'night', midnight: 'night city', clear: 'clear sky', cloudy: 'clouds', overcast: 'overcast', fog: 'fog', rain: 'rain', snow: 'snow', storm: 'storm',
  neon: 'neon', lamps: 'street lamps night', strings: 'christmas lights', candle: 'candles', moon: 'moonlight', sunrays: 'sun rays', citylights: 'city lights night', stage: 'stage lights' }
export const gapTerms = (gaps, n = 20) => gaps.filter((g) => /^(place|time|weather|light):/.test(g.key)).slice(0, n).map((g) => TERM[g.key.split(':')[1]]).filter(Boolean)
