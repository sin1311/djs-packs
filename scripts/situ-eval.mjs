// stdio 24장 V3-2d 상황 읽기 실측 (Actions · situ-eval): 모아 둔 실제 사진(scenes 가지 index.json 의 shots ~2,172장)을 앱과 같은 모델 · 같은 숫자로 읽고,
//  사진 제목(위키미디어 · Unsplash 원제목 + 한국어 꼬리표)에서 뽑은 느슨한 정답과 비교 → 축별 상위 1 · 3 적중률 · "모름" 문턱 쓸기 · 상황 77 대비.
//  느슨한 정답 = 제목에 그 요소를 뜻하는 낱말이 있을 때만(없으면 그 축은 채점 안 함). 결과 eval/situ-eval.json · eval/SITU_EVAL.md
import { writeFile, mkdir, readFile } from 'node:fs/promises'
const T = await import('@huggingface/transformers'), id = 'Xenova/siglip-base-patch16-224', N = +(process.env.N || 0), MB = 'https://raw.githubusercontent.com/sin1311/djs-packs/models/'
await mkdir('eval', { recursive: true })
const J = async (u) => { for (let i = 0; i < 4; i++) { try { const r = await fetch(u); if (r.ok) return r.json(); if (r.status === 404) return null } catch (e) { /* 다시 */ } await new Promise((s) => setTimeout(s, 800 * (i + 1))) } return null }
const idx = JSON.parse(await readFile('.scn/index.json', 'utf8')), FAC = await J(MB + 'siglip-fac.json'), SIT = await J(MB + 'siglip-situ.json'), FP = JSON.parse(await readFile('data/fac-prompts.json', 'utf8')).items, SP = JSON.parse(await readFile('data/situ-prompts.json', 'utf8')).items
const dec = (S) => S.int8.map((r) => Float32Array.from(r.q, (x) => x * r.s)), TF = dec(FAC), TS = dec(SIT), sig = (x) => 1 / (1 + Math.exp(-x))
/* 느슨한 정답: 요소 id → 제목 낱말(정규식). 영어 원제목 + 한국어 꼬리표 — 11-27 고침: '공연'(장르 꼬리표 = 공연장 건물) · 'ski'가 skies 에 걸리던 것 · '노래'(장르 꼬리표)는 정답에서 뺌 */
const K = {
  'place:beach': /\b(beach|seaside|shore|coast|sand)\b|바다|해변/i, 'place:mountain': /\b(mountains?|peak|alps|summit)\b|산\b/i, 'place:forest': /\b(forest|woods|woodland|trees)\b|숲/i, 'place:field': /\b(field|meadow|prairie)\b|들판/i,
  'place:lake': /\b(lake|river|pond|canal)\b|호수|강/i, 'place:snowland': /\b(snow|snowy|glacier)\b|눈 /i, 'place:desert': /\b(desert|dunes?)\b|사막/i, 'place:street': /\b(street|avenue|road in the city|downtown|crosswalk)\b|거리/i,
  'place:alley': /\b(alley|lane|backstreet)\b|골목/i, 'place:market': /\b(market|bazaar|stall)\b|시장/i, 'place:rooftop': /\b(rooftop|roof)\b|옥상/i, 'place:bridge': /\bbridge\b|다리/i, 'place:park': /\bpark\b|공원/i,
  'place:amusement': /\b(amusement|carousel|ferris|roller coaster|fairground)\b|놀이공원/i, 'place:stadium': /\b(stadium|arena)\b|경기장/i, 'place:stage': /\b(concert|stage|gig)\b|무대/i, 'place:club': /\b(club|nightclub|disco|rave)\b|클럽|디스코/i,
  'place:home': /\b(living room|home|house interior)\b|거실/i, 'place:kitchen': /\bkitchen\b|부엌/i, 'place:bedroom': /\bbedroom\b|침실/i, 'place:cafe': /\b(cafe|café|coffee)\b|카페/i, 'place:restaurant': /\b(restaurant|diner|bistro)\b|식당/i,
  'place:bar': /\b(bar|pub|tavern)\b|술집|바\b/i, 'place:office': /\boffice\b|사무실/i, 'place:classroom': /\b(classroom|school)\b|교실|학교/i, 'place:library': /\blibrary\b|도서관/i, 'place:church': /\b(church|cathedral|chapel|basilica)\b|성당|교회/i,
  'place:temple': /\b(temple|palace|shrine|pagoda)\b|절|고궁/i, 'place:museum': /\b(museum|gallery)\b|미술관|박물관/i, 'place:shop': /\b(mall|shop|store|boutique)\b|쇼핑/i, 'place:airport': /\b(airport|terminal)\b|공항/i, 'place:station': /\b(station|platform)\b|역/i,
  'place:train': /\b(train|railway|tram)\b|기차/i, 'place:car': /\b(car interior|dashboard)\b/i, 'place:road': /\b(road|highway|motorway)\b|도로/i, 'place:boat': /\b(boat|ship|ferry|yacht)\b|배/i, 'place:pool': /\b(swimming pool|pool)\b|수영장/i,
  'place:campsite': /\b(camping|campsite|tent|campfire)\b|캠핑/i, 'place:farm': /\b(farm|village|countryside|rural)\b|시골/i, 'place:garden': /\b(garden|flowers?|tulips?|roses?)\b|정원|꽃/i, 'place:wedding_hall': /\bwedding\b|결혼/i,
  'place:night_city': /\b(city at night|night city|skyline at night|neon)\b|네온|밤 도시|야경/i,
  'time:morning': /\b(morning|sunrise|dawn)\b|아침|해돋이/i, 'time:sunset': /\b(sunset|dusk|golden hour|twilight)\b|노을/i, 'time:night': /\b(night|evening|nocturne|neon)\b|밤|야경/i, 'time:rain': /\b(rain|rainy|umbrella)\b|비 /i,
  'time:snowfall': /\b(snowfall|snowing|blizzard)\b/i, 'time:fog': /\b(fog|foggy|mist|misty)\b|안개/i, 'time:storm': /\b(storm|thunder|lightning)\b|폭풍/i, 'time:spring': /\b(spring|cherry blossoms?|blossom)\b|봄|벚꽃/i,
  'time:summer': /\bsummer\b|여름/i, 'time:autumn': /\b(autumn|fall foliage|fall colors)\b|가을|단풍/i, 'time:winter': /\b(winter|christmas)\b|겨울/i,
  'activity:dancing': /\b(danc\w*|ballroom|waltz)\b|춤/i, 'activity:singing': /\b(singing|singer|singers|choir|karaoke)\b|합창/i, 'activity:playing_music': /\b(guitar\w*|piano\w*|band|musicians?|violin|orchestra|drummer)\b|연주/i, 'activity:running': /\b(running|marathon|runners?)\b|달리기/i,
  'activity:cycling': /\b(cycling|bicycle|bike)\b|자전거/i, 'activity:hiking': /\b(hiking|hikers?|trail)\b|등산/i, 'activity:surfing': /\bsurf\w*\b|서핑/i, 'activity:skiing': /\b(ski|skiing|skier|skiers|snowboard\w*)\b|스키/i,
  'activity:shopping': /\bshopping\b|쇼핑/i, 'activity:eating': /\b(eating|dinner|lunch|meal|feast)\b|식사/i, 'activity:cheering': /\b(cheering|fans|supporters)\b|응원/i, 'activity:marching': /\b(parade|march\w*)\b|행진/i, 'activity:performing': /\b(perform\w*|concert)\b/i,
  'event:christmas': /\b(christmas|xmas|advent)\b|성탄|크리스마스/i, 'event:wedding': /\bwedding\b|결혼/i, 'event:concert': /\b(concert|gig)\b|콘서트/i, 'event:festival': /\b(festival|carnival|fiesta)\b|축제|카니발/i, 'event:fireworks': /\bfireworks?\b|불꽃/i,
  'event:sports_match': /\b(match|game|football|soccer|baseball|basketball)\b|경기/i, 'event:marathon': /\bmarathon\b|마라톤/i, 'event:party': /\bparty\b|파티/i, 'event:newyear': /\bnew year\b|새해/i, 'event:halloween': /\bhalloween\b|할로윈/i, 'event:busking': /\b(busk\w*|street musicians?|street performer)\b|버스킹/i,
  'people:crowd': /\b(crowd|crowded|busy|people|festival|parade)\b|군중|사람들/i, 'people:couple': /\b(couple|lovers)\b|연인/i, 'people:family': /\bfamily\b|가족/i, 'people:kids': /\b(children|kids|child)\b|아이/i, 'people:alone': /\b(man|woman|girl|boy|portrait)\b/i,
}
const labelsOf = (title) => { const out = {}; for (const [k, re] of Object.entries(K)) if (re.test(title)) { const a = k.split(':')[0]; (out[a] = out[a] || []).push(k) } return out }
/* 77 상황도 같은 방식: 그 영어 문장의 핵심 낱말이 제목에 있으면 정답 */
const STOP = new Set(['a', 'an', 'the', 'with', 'and', 'of', 'in', 'on', 'at', 'by', 'or', 'to', 'people', 'photo', 'this', 'is', 'some', 'two', 'their', 'person'])
const situKw = SP.map((x) => x.en.toLowerCase().split(/[^a-z]+/).filter((w) => w.length > 3 && !STOP.has(w)))
const shots = (idx.shots || []).filter((s) => !s.synthetic && s.base), pick = N ? shots.filter((_, i) => i % Math.max(1, Math.floor(shots.length / N)) === 0).slice(0, N) : shots
console.log('사진', shots.length, '→ 재기', pick.length)
const proc = await T.AutoProcessor.from_pretrained(id), vq = await T.SiglipVisionModel.from_pretrained(id, { dtype: 'q8' })
const rows = []; let fail = 0, t0 = Date.now()
const one = async (s) => { const man = await J(s.base + s.manifest); if (!man || !man.color) { fail++; return } const img = await T.RawImage.fromURL(s.base + s.id + '/' + man.color).catch(() => null); if (!img) { fail++; return }
  const { pixel_values } = await proc(img.rgb()), e = (await vq({ pixel_values })).pooler_output.tolist()[0], n = Math.hypot(...e), u = e.map((x) => x / n)
  const cos = (TT) => TT.map((t) => t.reduce((a, v, k) => a + v * u[k], 0)), fp = cos(TF).map((c) => sig(FAC.scale * c + FAC.bias)), sp = cos(TS).map((c) => sig(SIT.scale * c + SIT.bias))
  const title = String(s.name || '').replace(/\((Unsplash|Wikimedia)[^)]*\)/gi, ''), lab = labelsOf(title), axes = {}
  for (const a of ['place', 'activity', 'event', 'people', 'time']) { const ks = FP.map((x, k) => [x.id, fp[k]]).filter(([i]) => i.startsWith(a + ':')).sort((x, y) => y[1] - x[1]); axes[a] = { top: ks.slice(0, 3).map(([i]) => i), conf: ks[0][1] } }
  const st = SP.map((x, k) => [k, sp[k]]).sort((x, y) => y[1] - x[1]).slice(0, 3), low = title.toLowerCase(), situHit = st.some(([k]) => situKw[k].some((w) => low.includes(w))), situLabeled = situKw.some((ws) => ws.some((w) => low.includes(w)))
  rows.push({ id: s.id, sub: s.sub, title: title.slice(0, 80), lab, axes, situ: { top: st.map(([k]) => SP[k].id), conf: st[0][1], hit: situHit, labeled: situLabeled } })
  if (rows.length % 100 === 0) console.log('…', rows.length, Math.round((Date.now() - t0) / 1000) + '초') }
for (let i = 0; i < pick.length; i += 6) await Promise.all(pick.slice(i, i + 6).map((s) => one(s).catch(() => { fail++ })))
/* 채점 */
const R = { n: rows.length, fail, axes: {}, situ77: {}, sweep: {} }
for (const a of ['place', 'activity', 'event', 'people', 'time']) { const L = rows.filter((r) => r.lab[a]), h1 = L.filter((r) => r.lab[a].includes(r.axes[a].top[0])).length, h3 = L.filter((r) => r.axes[a].top.some((x) => r.lab[a].includes(x))).length
  R.axes[a] = { labeled: L.length, top1: L.length ? +(h1 / L.length).toFixed(3) : null, top3: L.length ? +(h3 / L.length).toFixed(3) : null, confAll: q(rows.map((r) => r.axes[a].conf)) }
  R.sweep[a] = [0, 0.001, 0.002, 0.004, 0.006, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12].map((t) => { const K2 = L.filter((r) => r.axes[a].conf >= t), hit = K2.filter((r) => r.axes[a].top.some((x) => r.lab[a].includes(x))).length; return { t, keptLabeled: L.length ? +(K2.length / L.length).toFixed(3) : null, precision3: K2.length ? +(hit / K2.length).toFixed(3) : null, knownAll: +(rows.filter((r) => r.axes[a].conf >= t).length / Math.max(1, rows.length)).toFixed(3) } }) }
const L77 = rows.filter((r) => r.situ.labeled); R.situ77 = { labeled: L77.length, top3: L77.length ? +(L77.filter((r) => r.situ.hit).length / L77.length).toFixed(3) : null, confAll: q(rows.map((r) => r.situ.conf)) }
function q(xs) { const s = xs.slice().sort((a, b) => a - b), p = (f) => +s[Math.min(s.length - 1, Math.floor(f * s.length))].toPrecision(3); return s.length ? { p10: p(0.1), p50: p(0.5), p90: p(0.9), max: +s[s.length - 1].toPrecision(3) } : null }
await writeFile('eval/situ-eval.json', JSON.stringify({ at: new Date().toISOString(), R, rows }, null, 0))
const md = [`# 상황 읽기 실측 (${new Date().toISOString().slice(0, 10)} · 사진 ${R.n}장 · 못 읽음 ${fail})`, '', '느슨한 정답 = 사진 제목(원제목 + 한국어 꼬리표)에 그 요소 낱말이 있을 때만. 제목에 없는 요소는 채점 안 함 → 적중률은 아래로 치우친 값(제목이 다 말하지 않으니까).', '',
  '| 축 | 채점한 사진 | 상위 1 | 상위 3 | 확신(최고 sigmoid) p10 · p50 · p90 |', '|---|---|---|---|---|', ...Object.entries(R.axes).map(([a, v]) => `| ${a} | ${v.labeled} | ${v.top1} | ${v.top3} | ${v.confAll.p10} · ${v.confAll.p50} · ${v.confAll.p90} |`),
  `| 상황 77(통째) | ${R.situ77.labeled} | — | ${R.situ77.top3} | ${R.situ77.confAll.p10} · ${R.situ77.confAll.p50} · ${R.situ77.confAll.p90} |`, '', '## 문턱 쓸기(문턱 이상만 "앎" — 채점된 사진 중 남는 몫 · 남은 것의 상위 3 정확도 · 모든 사진 중 "앎" 몫)', '',
  ...Object.entries(R.sweep).map(([a, v]) => `- **${a}**: ` + v.map((x) => `${x.t} → 남음 ${x.keptLabeled} · 정확 ${x.precision3} · 앎 ${x.knownAll}`).join(' | '))].join('\n')
await writeFile('eval/SITU_EVAL.md', md + '\n'); console.log(md)
