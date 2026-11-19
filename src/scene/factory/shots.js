// 인서트 컷 후보 수집기 — 360° 장면이 아닌 "보통 사진" (멋진 차 · 사람들의 일상 · 곡 분위기에 맞는 순간)
//  · 주 흐름(실사 360° 장면)은 그대로이고, 컷은 섹션이 바뀔 때 가끔 몇 초 끼워 넣어요 (앱: src/three/scene/shot.js).
//  · 후보 = { src, id, kind: 'shot', sub: 'car' | 'people' | 'moment', title, author, license, source, url, thumb, w, h, text }
//  · 출처: Openverse(CC 사진 모음 · 열쇠 없이 · 플리커 등) + 위키미디어 공용 검색. 라이선스는 CC0 · PDM · CC BY · CC BY-SA(SA 팩)만.
//  · 사람 사진은 얼굴을 흐리고, 얼굴이 크게 나온 사진(가까운 인물 사진)은 장면 공장이 빼요 (scene-factory.mjs buildShot).
import { fetchRetry, wmThumb, isPano } from './sources.js'

export const SHOT_TERMS = {
  car: ['sports car night', 'classic car', 'vintage car street', 'car headlights night', 'car rain city', 'muscle car', 'motorcycle night', 'retro car neon', 'car sunset road', 'taxi night city'],
  people: ['street musician', 'people walking city night', 'festival crowd', 'concert crowd', 'cafe people', 'commuters subway', 'night market people', 'skateboarding', 'people dancing', 'friends beach sunset', 'umbrella rain street', 'cyclist city'],
  moment: ['neon sign', 'vinyl record', 'rain window', 'city lights bokeh', 'coffee cup cafe', 'fireworks', 'christmas lights', 'turntable dj', 'sparkler', 'lanterns night'],
}
export const isShotShape = (w, h) => w > 0 && h > 0 && w / h >= 1.3 && w / h <= 2.4 && !isPano(w, h) /* 가로 사진만 (화면을 덮어요) · 2:1 등장방형 파노라마는 360° 장면 쪽 */
const ovLicense = (r) => { const l = String(r.license || '').toLowerCase(), v = r.license_version ? ' ' + r.license_version : ''; return l === 'cc0' ? 'CC0' : l === 'pdm' ? 'PDM' : l === 'by' ? 'CC BY' + v : l === 'by-sa' ? 'CC BY-SA' + v : 'CC ' + l.toUpperCase() + v }

// Openverse: 열쇠 없는(익명) 요청은 분당 몇 번으로 제한돼서 검색어마다 한 쪽(20장)씩, 요청 사이 3.2초 (sources.js 호스트 간격)
export async function openverse(fetch, { terms = SHOT_TERMS, pages = 1, minW = 1000, wait } = {}) {
  const out = [], seen = new Set()
  for (const [sub, list] of Object.entries(terms)) for (const t of list) for (let pg = 1; pg <= pages; pg++) {
    const q = new URLSearchParams({ q: t, license: 'cc0,pdm,by,by-sa', category: 'photograph', aspect_ratio: 'wide', size: 'large', mature: 'false', page_size: '20', page: String(pg) })
    let j; try { j = await (await fetchRetry('https://api.openverse.org/v1/images/?' + q, { fetch, tries: 3, ...(wait ? { wait } : {}) })).json() } catch (e) { console.log('[mass] openverse', t, '건너뜀:', String(e.message).slice(0, 80)); break }
    for (const r of j.results || []) {
      if (!r.id || seen.has(r.id) || !r.url) continue; seen.add(r.id)
      const w = +r.width || 0, h = +r.height || 0; if (w && h && (!isShotShape(w, h) || w < minW)) continue
      out.push({ src: 'openverse', kind: 'shot', sub, id: 'ov-' + r.id, title: String(r.title || t).slice(0, 80), author: r.creator || '', license: ovLicense(r), source: r.foreign_landing_url || '', url: r.url, thumb: r.url, w: w || 1024, h: h || 640, text: [t, r.title, ...(r.tags || []).map((x) => x.name)].join(' ').slice(0, 400) })
    }
    if (!j.page_count || pg >= j.page_count) break
  }
  return out
}
// 위키미디어 공용: 검색어마다 보통(평면) 사진 · 가로 2400 이상 (화질이 좋아요)
export async function wikimediaShots(fetch, { terms = SHOT_TERMS, per = 40, minW = 2400, wait } = {}) {
  const out = [], seen = new Set(), strip = (h) => String(h || '').replace(/<[^>]+>/g, ' ').replace(/&[a-z]+;/g, ' ').replace(/\s+/g, ' ').trim()
  for (const [sub, list] of Object.entries(terms)) for (const t of list) {
    const q = new URLSearchParams({ action: 'query', format: 'json', generator: 'search', gsrsearch: `${t} filetype:bitmap -panorama -equirectangular`, gsrnamespace: '6', gsrlimit: String(Math.min(50, per)), prop: 'imageinfo', iiprop: 'url|size|mime|extmetadata', iiurlwidth: '1920', origin: '*' })
    let j; try { j = await (await fetchRetry('https://commons.wikimedia.org/w/api.php?' + q, { fetch, tries: 3, ...(wait ? { wait } : {}) })).json() } catch (e) { console.log('[mass] wikimedia 컷', t, '건너뜀:', String(e.message).slice(0, 80)); continue }
    for (const p of Object.values((j.query || {}).pages || {})) {
      const ii = (p.imageinfo || [])[0]; if (!ii || seen.has(p.pageid)) continue; seen.add(p.pageid)
      if (!/jpeg|png/.test(ii.mime || '') || !isShotShape(ii.width, ii.height) || ii.width < minW) continue
      const m = ii.extmetadata || {}, v = (k) => strip((m[k] || {}).value)
      out.push({ src: 'wikimedia', kind: 'shot', sub, id: 'wms-' + p.pageid, title: p.title.replace(/^File:/, '').replace(/\.\w+$/, '').slice(0, 80), author: v('Artist') || '', license: v('LicenseShortName') || v('License'), source: ii.descriptionurl, url: wmThumb(ii.thumburl || ii.url, 1920), thumb: wmThumb(ii.thumburl || ii.url, 1280), w: ii.width, h: ii.height, text: [t, p.title, v('ImageDescription')].join(' ').slice(0, 400) })
    }
  }
  return out
}
export const SHOT_SOURCES = { openverse, wikimediaShots }
// 선별 몫: 컷은 주 흐름이 아니라서 전체의 일부만 (기본 8%) · 갈래(차 · 사람 · 순간)가 고루
export function shotQuota(total, share = 0.08) { return Math.max(0, Math.round(total * share)) }
