// v1.0-D 장면 후보 수집기 (출처 5곳) — 후보 = { src, id, title, author, license, source(출처 쪽), url(원본), thumb(≤2048), w, h, text(제목 · 태그), sky?, geo? }
//  · 사진 자체는 아직 받지 않아요(목록만). 품질 관문(quality.js)이 작은 사본을 받아 재요.
//  · 이 저장소 환경에서는 출처 사이트에 접속할 수 없어서 응답 형식은 시험 자료로만 확인했어요 [확인 필요: Actions 로그].
//  · 모든 수집기는 fetch 를 주입받아요(시험에서 흉내). 실패하면 그 출처만 건너뛰어요.
//  · 위키미디어 정책: 자동 요청에는 연락 가능한 User-Agent 가 필요해요 → 프로젝트 저장소 주소만 넣어요(개인 정보는 넣지 않아요).
export const UA = 'djs-scene-factory/1.0 (+https://github.com/sin1311/djs-packs)'
export const redact = (u) => String(u).replace(/(access_token|token|key)=[^&]*/gi, '$1=***') /* 공개 로그에 토큰이 찍히지 않게 (첫 실행에서 주소째 찍혔어요) */
const J = async (fetch, url, o = {}) => { const r = await fetch(url, { headers: { 'User-Agent': UA, Accept: 'application/json', ...(o.headers || {}) } }); if (!r.ok) throw new Error(`HTTP ${r.status} ${redact(url).slice(0, 80)}`); return r.json() }
const strip = (h) => String(h || '').replace(/<[^>]+>/g, ' ').replace(/&[a-z]+;/g, ' ').replace(/\s+/g, ' ').trim()
export const isPano = (w, h) => w > 0 && h > 0 && Math.abs(w / h - 2) < 0.04

// 1) Poly Haven (CC0) — HDRI 전부. 하늘만 있는 것(skies)은 하늘 도서관으로 표시
export async function polyhaven(fetch, { limit = 2000 } = {}) {
  const all = await J(fetch, 'https://api.polyhaven.com/assets?t=hdris'), out = []
  for (const [id, a] of Object.entries(all)) {
    const cats = (a.categories || []).map((x) => String(x).toLowerCase()), sky = cats.includes('skies') && !cats.some((c) => /urban|indoor/.test(c)) /* 하늘 HDRI 에는 natural light 태그가 붙어요 (첫 실행에서 하늘 0개였던 원인) */
    out.push({ src: 'polyhaven', id: 'ph-' + id, title: a.name || id, author: Object.keys(a.authors || {}).join(', ') || 'Poly Haven', license: 'CC0', source: 'https://polyhaven.com/a/' + id, url: `https://dl.polyhaven.org/file/ph-assets/HDRIs/extra/Tonemapped%20JPG/${id}.jpg`, thumb: `https://cdn.polyhaven.com/asset_img/thumbs/${id}.png?width=1024&height=512`, w: (a.max_resolution || [8192])[0], h: (a.max_resolution || [0, 4096])[1], text: [a.name, ...cats, ...(a.tags || [])].join(' '), sky })
    if (out.length >= limit) break
  }
  return out
}
// 2) ambientCG (CC0) — HDRI 목록 (내려받기 형식이 여러 가지라 JPG/PNG 미리보기를 우선) [확인 필요]
export async function ambientcg(fetch, { limit = 1000 } = {}) {
  const out = []
  for (let off = 0; off < limit; off += 100) {
    const j = await J(fetch, `https://ambientcg.com/api/v2/full_json?type=HDRI&include=downloadData,tagData,imageData&limit=100&offset=${off}`), list = j.foundAssets || []
    if (off === 0) console.log('[mass] ambientcg 응답 모양:', JSON.stringify(Object.keys(j)).slice(0, 200), list[0] ? JSON.stringify(list[0]).slice(0, 600) : '(목록 없음)') /* 첫 실행에서 0개 → 형식 확인용 */
    for (const a of list) {
      const dl = Object.values(a.downloadFolders || {}).flatMap((f) => Object.values(f.downloadFiletypeCategories || {}).flatMap((c) => c.downloads || [])), pick = (re) => dl.filter((d) => re.test(d.attribute || d.fileName || '')).sort((x, y) => (x.size || 0) - (y.size || 0))[0]
      const img = pick(/TONEMAPPED.*(JPG|PNG)|8K-JPG|4K-JPG/i) || pick(/JPG|PNG/i); if (!img) continue
      const prev = (a.previewImage || {})['1024-PNG'] || (a.previewImage || {})['512-PNG'] || img.downloadLink
      out.push({ src: 'ambientcg', id: 'acg-' + a.assetId, title: a.displayName || a.assetId, author: 'ambientCG', license: 'CC0', source: 'https://ambientcg.com/view?id=' + a.assetId, url: img.downloadLink || img.fullDownloadPath, thumb: prev, w: 8192, h: 4096, text: [a.displayName, ...(a.tags || [])].join(' '), sky: (a.tags || []).some((t) => /^sky|skies|clouds?$/i.test(t)) })
    }
    if (list.length < 100) break
  }
  return out
}
// 3) 위키미디어 공용 — 등장방형 360° 파노라마 (파일마다 라이선스). 분위기 칸을 채울 검색어로 여러 번
export const WM_CATS = ['Category:Equirectangular projections', 'Category:360° panoramas', 'Category:Spherical panoramas', 'Category:Equirectangular panoramas']
export async function wikimedia(fetch, { terms = ['equirectangular'], per = 200, minW = 4000, cats = WM_CATS, perCat = 1500 } = {}) {
  const out = [], seen = new Set()
  const take = (p) => { const ii = (p.imageinfo || [])[0]; if (!ii || seen.has(p.pageid)) return; seen.add(p.pageid); if (!/jpeg|png/.test(ii.mime || '') || !isPano(ii.width, ii.height) || ii.width < minW) return; const m = ii.extmetadata || {}, v = (k) => strip((m[k] || {}).value); out.push({ src: 'wikimedia', id: 'wm-' + p.pageid, title: p.title.replace(/^File:/, '').replace(/\.\w+$/, ''), author: v('Artist') || '?', license: v('LicenseShortName') || v('License'), source: ii.descriptionurl, url: ii.thumburl || ii.url, thumb: (ii.thumburl || ii.url || '').replace(/\/\d+px-/, '/1024px-'), w: ii.width, h: ii.height, text: [p.title, v('ImageDescription'), v('Categories')].join(' ').slice(0, 400) }) }
  for (const c of cats) { let cont = {}; for (let got = 0; got < perCat;) { let j; try { j = await J(fetch, 'https://commons.wikimedia.org/w/api.php?' + new URLSearchParams({ action: 'query', format: 'json', generator: 'categorymembers', gcmtitle: c, gcmtype: 'file', gcmlimit: '50', prop: 'imageinfo', iiprop: 'url|size|mime|extmetadata', iiurlwidth: '4096', origin: '*', ...cont })) } catch (e) { console.log('[mass] wikimedia', c, '건너뜀:', String(e.message).slice(0, 60)); break } const pages = Object.values((j.query || {}).pages || {}); for (const p of pages) { take(p); got++ } if (!j.continue || !pages.length) break; cont = j.continue } } /* 360° 파노라마 분류에서 먼저 (검색어만으로는 164개뿐이었어요) */
  for (const t of terms) {
    let cont = {}
    for (let got = 0; got < per;) {
      const q = new URLSearchParams({ action: 'query', format: 'json', generator: 'search', gsrsearch: `${t} equirectangular filetype:bitmap`, gsrnamespace: '6', gsrlimit: '50', prop: 'imageinfo', iiprop: 'url|size|mime|extmetadata', iiurlwidth: '4096', origin: '*', ...cont })
      let j; try { j = await J(fetch, 'https://commons.wikimedia.org/w/api.php?' + q) } catch (e) { break } const pages = Object.values((j.query || {}).pages || {})
      for (const p of pages) {
        const ii = (p.imageinfo || [])[0]; if (!ii || seen.has(p.pageid)) continue; seen.add(p.pageid); got++
        if (!/jpeg|png/.test(ii.mime || '') || !isPano(ii.width, ii.height) || ii.width < minW) continue
        const m = ii.extmetadata || {}, v = (k) => strip((m[k] || {}).value)
        out.push({ src: 'wikimedia', id: 'wm-' + p.pageid, title: p.title.replace(/^File:/, '').replace(/\.\w+$/, ''), author: v('Artist') || '?', license: v('LicenseShortName') || v('License'), source: ii.descriptionurl, url: ii.thumburl || ii.url, thumb: (ii.thumburl || ii.url || '').replace(/\/\d+px-/, '/1024px-'), w: ii.width, h: ii.height, text: [p.title, v('ImageDescription'), v('Categories')].join(' ').slice(0, 400) })
      }
      if (!j.continue || !pages.length) break; cont = j.continue
    }
  }
  return out
}
// 4) Panoramax (공개 거리 사진 · 얼굴 · 번호판은 플랫폼이 흐림) — 도시 영역마다 STAC 검색, 시야 360° 만
export async function panoramax(fetch, { bboxes = CITY_BBOX, per = 60 } = {}) {
  const out = []
  for (const [name, bb] of bboxes) {
    let j; try { j = await J(fetch, `https://api.panoramax.xyz/api/search?bbox=${bb.join(',')}&limit=${per}`) } catch (e) { console.log('[mass] panoramax', name, '건너뜀:', String(e.message).slice(0, 80)); continue } /* 도시 하나가 실패해도 나머지는 계속 */
    for (const f of j.features || []) {
      const pr = f.properties || {}, io = pr['pers:interior_orientation'] || {}, a = f.assets || {}; if (+io.field_of_view !== 360) continue
      const W = +(pr['pers:interior_orientation'] || {}).sensor_array_dimensions?.[0] || 5760, H = +(pr['pers:interior_orientation'] || {}).sensor_array_dimensions?.[1] || 2880
      out.push({ src: 'panoramax', id: 'px-' + f.id, title: `${name} 거리 (${(pr.datetime || '').slice(0, 10)})`, author: (f.providers || pr.providers || []).map((p) => p.name).join(', ') || 'Panoramax 기여자', license: pr.license || 'etalab-2.0', source: `https://api.panoramax.xyz/#focus=pic&pic=${f.id}`, url: (a.hd || a.sd || {}).href, thumb: (a.sd || a.thumb || {}).href, w: W, h: H, text: `${name} street ${pr.datetime || ''}`, geo: f.geometry && f.geometry.coordinates, blurred: true })
    }
  }
  return out
}
// 5) Mapillary (CC BY-SA 4.0 → SA 팩 · 무료 토큰 · 플랫폼이 흐림) — 도시 영역마다 is_pano
export async function mapillary(fetch, { token, bboxes = CITY_BBOX, per = 60 } = {}) {
  if (!token) return []
  const out = []
  for (const [name, bb0] of bboxes) {
    const cx = (bb0[0] + bb0[2]) / 2, cy = (bb0[1] + bb0[3]) / 2, bb = [cx - 0.006, cy - 0.006, cx + 0.006, cy + 0.006] /* 작은 영역 (큰 영역은 서버 오류 500 이 났어요) */
    let j; try { j = await J(fetch, `https://graph.mapillary.com/images?access_token=${encodeURIComponent(token)}&fields=id,thumb_2048_url,thumb_1024_url,is_pano,creator&is_pano=true&bbox=${bb.join(',')}&limit=${per}`) } catch (e) { console.log('[mass] mapillary', name, '건너뜀:', String(e.message).slice(0, 80)); continue }
    for (const im of j.data || []) { if (!im.is_pano) continue; out.push({ src: 'mapillary', id: 'mly-' + im.id, title: `${name} 거리`, author: (im.creator || {}).username || 'Mapillary 기여자', license: 'CC-BY-SA-4.0', source: 'https://www.mapillary.com/app/?pKey=' + im.id, url: im.thumb_2048_url, thumb: im.thumb_1024_url || im.thumb_2048_url, w: im.width || 4096, h: im.height || 2048, text: `${name} street`, geo: null, blurred: true }) }
  }
  return out
}
// 거리 사진을 찾을 도시 영역 [서경, 남위, 동경, 북위] (작은 도심 상자)
export const CITY_BBOX = [
  ['파리', [2.33, 48.85, 2.36, 48.87]], ['리옹', [4.82, 45.75, 4.85, 45.77]], ['마르세유', [5.36, 43.29, 5.39, 43.31]], ['보르도', [-0.59, 44.83, -0.56, 44.85]], ['낭트', [-1.57, 47.21, -1.54, 47.23]], ['스트라스부르', [7.74, 48.57, 7.76, 48.59]], ['니스', [7.26, 43.69, 7.28, 43.71]], ['툴루즈', [1.43, 43.59, 1.46, 43.61]],
  ['브뤼셀', [4.34, 50.84, 4.37, 50.86]], ['암스테르담', [4.88, 52.36, 4.91, 52.38]], ['베를린', [13.39, 52.51, 13.42, 52.53]], ['바르셀로나', [2.16, 41.38, 2.19, 41.40]], ['리스본', [-9.15, 38.70, -9.12, 38.72]], ['로마', [12.47, 41.89, 12.50, 41.91]], ['프라하', [14.41, 50.08, 14.44, 50.09]], ['빈', [16.36, 48.20, 16.38, 48.22]],
  ['런던', [-0.14, 51.50, -0.11, 51.52]], ['뉴욕', [-74.0, 40.71, -73.97, 40.76]], ['도쿄', [139.69, 35.65, 139.72, 35.70]], ['서울', [126.97, 37.55, 127.0, 37.57]], ['홍콩', [114.15, 22.27, 114.18, 22.29]], ['싱가포르', [103.84, 1.28, 103.86, 1.30]], ['시드니', [151.20, -33.88, 151.22, -33.86]], ['멕시코시티', [-99.15, 19.42, -99.12, 19.44]],
]
export const SOURCES = { polyhaven, ambientcg, wikimedia, panoramax, mapillary }
