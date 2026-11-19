// v1.0-C 장면 자산 라이선스 검사 (장면 공장 · 앱 공용)
//  · 받는 것: CC0 · 퍼블릭 도메인(PDM) · CC BY (2.0/3.0/4.0, 저작자 · 출처 주소 필수).
//  · CC BY-SA 는 "SA 팩"에만 (불변: SA 는 섞지 않아요 — v1.0-A 라이선스 SA 규칙과 같음).
//  · 비영리(NC) · 변경 금지(ND) · 모름 은 거절해요. 구글 스트리트뷰는 약관상 출처 자체를 받지 않아요.
const OK = /^(cc0|cc0-1\.0|pdm|public ?domain|cc[- ]by(-[234]\.0)?)$/i
export function normLicense(s) {
  const t = String(s || '').trim().toLowerCase().replace(/_/g, '-').replace(/^https?:\/\/creativecommons\.org\/(licenses|publicdomain)\//, '').replace(/\/$/, '')
  if (/^zero(\/1\.0)?$|^cc0/.test(t)) return 'CC0'
  if (/^mark(\/1\.0)?$|^pdm$|public ?domain/.test(t)) return 'PDM'
  if (/etalab|licence ouverte|open licence|^lo-?2(\.0)?$/.test(t)) return 'ETALAB-2.0' /* 프랑스 공개 라이선스 2.0: 저작자 표기 조건 (CC BY 와 호환) */
  const m = t.match(/^(?:cc[- ])?(by(?:-sa|-nc|-nd|-nc-sa|-nc-nd)?)[-/ ]?([234]\.0)?/); if (m) return ('CC-' + m[1].toUpperCase() + (m[2] ? '-' + m[2] : ''))
  return t ? t.toUpperCase() : 'UNKNOWN'
}
// → { ok, pack: 'main' | 'sa' | null, why }
export function checkLicense(a) {
  const L = normLicense(a.license), src = String(a.source || '')
  if (/google\.[a-z.]+\/maps|streetview|street-view/i.test(src)) return { ok: false, pack: null, why: '스트리트뷰: 약관상 수집 · 재사용 금지', L }
  if (/NC|ND/.test(L.replace(/^CC-/, ''))) return { ok: false, pack: null, why: '비영리 · 변경 금지 라이선스', L }
  if (L === 'CC0' || L === 'PDM') return { ok: true, pack: 'main', why: '', L }
  if (L === 'ETALAB-2.0') return a.author && src ? { ok: true, pack: 'main', why: '', L } : { ok: false, pack: null, why: 'Etalab 2.0: 저작자 · 출처 주소가 필요해요', L }
  if (/^CC-BY(-[234]\.0)?$/.test(L)) return a.author && src ? { ok: true, pack: 'main', why: '', L } : { ok: false, pack: null, why: 'CC BY: 저작자 · 출처 주소가 필요해요', L }
  if (/^CC-BY-SA/.test(L)) return a.author && src ? { ok: true, pack: 'sa', why: 'SA 팩에만', L } : { ok: false, pack: null, why: 'CC BY-SA: 저작자 · 출처 주소가 필요해요', L }
  return { ok: false, pack: null, why: '모르는 라이선스: ' + L, L }
}
export const attribution = (a) => (a.license && normLicense(a.license) !== 'CC0' ? `${a.title || a.id} — ${a.author || '?'} (${normLicense(a.license)}) ${a.source || ''}`.trim() : `${a.title || a.id} (CC0)`)
export { OK as LICENSE_OK }
