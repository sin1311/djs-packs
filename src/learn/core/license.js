// 라이선스 · 출처 · 사생활 검사 (브라우저 · 서버 · 크론 공용). 자동 수집한 자산은 이 검사를 통과해야만 앱에 들어와요.
//  · 모든 자산: id · 종류 · 출처 · 주소(https) · 라이선스 · 저작자 기록 필수
//  · 등급 5가지: open(퍼블릭 도메인 · CC0 · MIT …) · attribution(CC BY — 저작자 표기) · sharealike(CC BY-SA · ODbL — 표기 + 같은 조건 공유)
//               · limited(비상업 · 매출 제한 · 카피레프트 — 상업 배포 거절) · unknown(거절)
//  · 기본값 = 상업 배포 안전 모드 (commercial: true). 개인 · 연구용으로만 쓸 때만 commercial: false 로 경고만 받아요.
//  · 같은 조건 공유(SA) 자산은 "SA 팩"에만 — 기본 팩(core)에 섞으면 그 팩 전체가 SA 조건을 물려받으니 거절해요.
//    예외: 통계만 뽑고 원문을 배포하지 않는 데이터(use: 'stats') — 사실 · 숫자에는 SA 조건이 따라오지 않아 경고만.
//  · 사생활: 실제 거리 · 사람이 찍힌 이미지 · 파노라마 · 영상은 얼굴 · 번호판 흐림 처리 기록이 있어야 통과
export const OPEN = { 'PD': '퍼블릭 도메인', 'CC0-1.0': 'CC0', 'MIT': 'MIT', 'Apache-2.0': 'Apache 2.0', 'BSD-2-Clause': 'BSD', 'BSD-3-Clause': 'BSD', 'OFL-1.1': 'SIL OFL', 'ISC': 'ISC', 'Zlib': 'zlib' }
export const ATTRIB = { 'CC-BY-4.0': 'CC BY 4.0', 'CC-BY-3.0': 'CC BY 3.0', 'CC-BY-2.0': 'CC BY 2.0' }
export const SHAREALIKE = { 'CC-BY-SA-4.0': 'CC BY-SA 4.0', 'CC-BY-SA-3.0': 'CC BY-SA 3.0', 'CC-BY-SA-2.0': 'CC BY-SA 2.0', 'ODbL-1.0': 'ODbL' }
export const LIMITED = { 'CC-BY-NC-4.0': '비상업 전용', 'CC-BY-NC-SA-4.0': '비상업 전용', 'CC-BY-NC-3.0': '비상업 전용', 'OpenRAIL': '용도 제한 조항', 'OpenRAIL-M': '용도 제한 조항', 'Stability-Community': '연 매출 제한', 'AGPL-3.0': '네트워크 공개 의무(카피레프트)', 'GPL-3.0': '카피레프트', 'Custom-Review': '개별 검토 필요' }
export const LEVELS = [['open', '자유'], ['attribution', '저작자 표기'], ['sharealike', '같은 조건 공유 (SA 팩)'], ['limited', '제한 (상업 배포 불가)'], ['unknown', '알 수 없음']]
export const KINDS = ['code', 'font', 'data', 'midi', 'audio', 'sample', 'ir', 'image', 'pano', 'hdri', 'model3d', 'splat', 'video', 'model', 'service']
export const PACKS = { core: '기본 팩', sa: 'SA 팩 (CC BY-SA 로 따로 배포)' }
const STREET = new Set(['image', 'pano', 'video', 'splat'])
const one = (lic) => (OPEN[lic] ? 'open' : ATTRIB[lic] ? 'attribution' : SHAREALIKE[lic] ? 'sharealike' : LIMITED[lic] ? 'limited' : 'unknown')
const RANK = ['open', 'attribution', 'sharealike', 'limited', 'unknown']
// SPDX 선택 표현 "A OR B"(이중 라이선스 — 받는 쪽이 고를 수 있어요): 아는 것 중 가장 덜 제한적인 쪽으로 (AND · WITH 는 안 받아요 → 모름)
export function classify(lic) { const alts = String(lic || '').replace(/^\(|\)$/g, '').split(/\s+OR\s+/); if (alts.length === 1) return one(lic); return alts.map((x) => one(x.trim())).sort((a, b) => RANK.indexOf(a) - RANK.indexOf(b))[0] }
const name1 = (lic) => OPEN[lic] || ATTRIB[lic] || SHAREALIKE[lic] || LIMITED[lic] || lic
export const licName = (lic) => String(lic || '').split(/\s+OR\s+/).map((x) => name1(x.trim())).join(' 또는 ')
// a: { id, kind, title, source, url, license, author, pack: 'core'|'sa', use: 'stats'|'ship', tags: [], privacy: { faces, plates } }
export function validateAsset(a, { commercial = true, pack } = {}) {
  const problems = [], warns = []
  if (!a || typeof a !== 'object') return { ok: false, level: 'unknown', problems: ['자산 정보가 없어요'], warns }
  for (const k of ['id', 'kind', 'source', 'license']) if (!a[k]) problems.push(k + ' 없음')
  if (a.kind && !KINDS.includes(a.kind)) problems.push('알 수 없는 종류: ' + a.kind)
  if (a.url && !/^https:\/\//.test(a.url)) problems.push('주소는 https 만')
  const level = classify(a.license), pk = pack || a.pack || 'core'
  if (!PACKS[pk]) problems.push('알 수 없는 팩: ' + pk)
  if (level === 'unknown') problems.push('허용 목록에 없는 라이선스: ' + (a.license || '없음'))
  if ((level === 'attribution' || level === 'sharealike') && !(a.author && a.author !== 'unknown')) problems.push('저작자 표기가 필요한 라이선스인데 저작자가 없어요')
  if (level === 'sharealike') {
    if (a.use === 'stats') warns.push('같은 조건 공유 — 통계만 쓰고 원문은 배포하지 않아요')
    else if (pk !== 'sa') problems.push('같은 조건 공유(SA) 자산은 기본 팩에 섞을 수 없어요 — SA 팩으로 따로 배포')
    else warns.push('SA 팩: 이 팩과 이걸로 만든 결과물은 CC BY-SA 로 공유돼요')
  }
  if (level === 'limited') { if (commercial) problems.push(LIMITED[a.license] + ' — 상업 배포에는 쓸 수 없어요'); else warns.push(LIMITED[a.license] + ' — 개인 · 연구용으로만') }
  if (pk === 'sa' && level === 'limited') problems.push('SA 팩에도 비상업 · 제한 라이선스는 넣을 수 없어요')
  if (STREET.has(a.kind) && (a.tags || []).some((t) => /street|city|people|crowd|road|market|거리|도시|사람|시장|도로/.test(t))) {
    const pv = a.privacy || {}; if (!['blurred', 'none'].includes(pv.faces)) problems.push('사람이 나올 수 있는 거리 자산: 얼굴 흐림 처리 기록 필요'); if (!['blurred', 'none'].includes(pv.plates)) problems.push('거리 자산: 번호판 흐림 처리 기록 필요')
  }
  return { ok: problems.length === 0, level, pack: pk, problems, warns }
}
// 팩 단위 검사: 팩 안 모든 자산 + 섞임 규칙. SA 팩은 팩 자체 라이선스가 CC BY-SA 여야 해요 (CC0 · CC BY 는 SA 팩 안에 들어갈 수 있어요 — 반대는 안 돼요).
export function validatePack(p, { commercial = true } = {}) {
  const problems = [], items = (p && p.assets) || [], pk = (p && p.pack) || 'core'
  if (!p || !p.id) problems.push('팩 id 없음')
  if (pk === 'sa' && !SHAREALIKE[p.license]) problems.push('SA 팩은 팩 라이선스를 CC BY-SA 로 적어야 해요')
  if (pk === 'core' && p.license && classify(p.license) === 'sharealike') problems.push('기본 팩에 SA 라이선스를 붙일 수 없어요')
  const res = items.map((a) => ({ id: a.id, ...validateAsset(a, { commercial, pack: pk }) }))
  for (const r of res) if (!r.ok) problems.push(r.id + ': ' + r.problems.join(', '))
  return { ok: problems.length === 0, pack: pk, problems, items: res, credits: items.map(attribution) }
}
export function attribution(a) { const lv = classify(a.license); return `${a.title || a.id}${a.author && a.author !== 'unknown' ? ' — ' + a.author : ''} (${licName(a.license)}${lv === 'sharealike' && a.use !== 'stats' ? ' · 같은 조건 공유' : ''}, ${a.source})` }
