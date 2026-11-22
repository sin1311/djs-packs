/* 사진 장면 큐레이션 — 장르 브리프 · 감성(멋짐 · 낭만 · 추억) · 불쾌/이상 거절 (2026-11-21 사용자 지시)
   사용자: "여행을 갔을 때 그 도시와 어울리는 노래를 듣는 상황을 거꾸로 — 노래에 맞는 도시에 가거나 활동을 한다고 생각해 봐.
           캐롤 = 유럽 크리스마스 마켓 · 장식된 길거리, 힙합 = 뉴욕 어두운 밤 골목 · 거친 피사체, 시티팝 = 노을 진 대도시 · 올드카 안에서 본 거리.
           잘 어울려도 불쾌감 · 이상한 느낌을 줄 사진은 넣지 말고, 멋짐 · 낭만 · 추억 같은 좋은 감정을 불러일으킬 사진만."
   - 장면 공장(Actions)이 CLIP(OpenAI CLIP ViT-B/32 · MIT, Xenova ONNX)으로 이 글들을 장면의 원근 4장과 비교해요 (scene-mass.mjs rescore · gate).
   - 앱은 결과(장면마다 g = 장르 적합도)만 읽어 장르에 맞는 장면을 골라요 (select.js).
   - 순수 데이터 + 순수 함수 (DOM · 모델 없음). 시험: unit37.
   - 2026-11-21 둘째 지시: 가까이 찍은 사진 빼기 · 작품으로 남길 만한 사진만 → craft.js (미학 점수 · 전문/아마추어 · 넓은/가까이 · 화소 통계)를 판정에 더해요. */
import { CRAFT_PROMPTS, isCraftKey, axis, aesScore, techVerdict, TECH_CUT } from './craft.js'

/* 장르 브리프: want = 사람이 읽는 한 줄 · prompts = CLIP 비교 문장(영어) · terms = 360° 파노라마 검색어(위키미디어 · 파노라막스)
   shots = 인서트 컷(보통 사진) 검색어(Openverse) · places = 명소 이름(검색어에 덧붙여요) */
export const BRIEFS = {
  carol: { ko: '캐롤 · 홀리데이', want: '유럽 크리스마스 마켓 · 전구와 장식으로 꾸민 길거리 · 눈 내린 구시가 광장',
    prompts: ['a festive european christmas market at night with warm lights and wooden stalls', 'a beautiful street decorated with christmas lights and garlands', 'a snowy old town square with a big christmas tree', 'a cozy winter night in a christmas village with fairy lights'],
    terms: ['Christmas market', 'Weihnachtsmarkt', 'Marché de Noël', 'Christkindlmarkt', 'Christmas lights street', 'Christmas tree square'], shots: ['christmas market night lights', 'christmas lights street', 'christmas tree city square night', 'christmas decorations shop window'],
    places: ['Strasbourg', 'Nuremberg', 'Vienna', 'Prague', 'Cologne', 'Dresden', 'Colmar', 'Tallinn'] },
  waltz: { ko: '왈츠', want: '빈 · 파리 같은 우아한 유럽 구시가 · 궁전 정원 · 오페라 극장 앞 광장 · 노을 진 강변',
    prompts: ['an elegant european old town square with classical buildings at dusk', 'a grand palace garden with fountains', 'a romantic riverside promenade in an old european city at sunset', 'a beautiful opera house plaza at night'],
    terms: ['Schönbrunn', 'Vienna old town', 'palace garden', 'opera house square', 'Paris Seine'], shots: ['vienna old town evening', 'palace garden fountain', 'paris seine sunset'], places: ['Vienna', 'Paris', 'Budapest', 'Salzburg', 'Prague'] },
  citypop: { ko: '시티팝', want: '노을 진 대도시 거리 · 올드카 안에서 본 도시 · 해안 도로 · 야자수 대로 · 황혼의 네온',
    prompts: ['a big city street at sunset with golden light and tall buildings', 'the view from inside a vintage car driving through a city at dusk', 'a coastal highway at sunset with palm trees', 'a retro city street at twilight with glowing neon signs'],
    terms: ['sunset street', 'city at dusk', 'Shibuya', 'Ginza', 'coastal road sunset', 'boulevard palm trees'], shots: ['vintage car city sunset', 'classic car interior view city', 'tokyo street dusk', 'city sunset street golden hour', 'palm trees boulevard sunset'],
    places: ['Tokyo', 'Los Angeles', 'Miami', 'Hong Kong', 'New York', 'Seoul'] },
  trap: { ko: '트랩', want: '어둡고 거친 대도시의 밤 · 네온이 비친 젖은 거리 · 야경 고층 빌딩 · 밤의 주차 타워 옥상',
    prompts: ['a dark moody city street at night with neon reflections on wet asphalt', 'a gritty downtown at night under street lights', 'a city skyline at night seen from a rooftop', 'an urban street at night with a dramatic cinematic atmosphere'],
    terms: ['night street', 'downtown night', 'rooftop night skyline', 'Times Square night', 'Atlanta night'], shots: ['dark city street night neon', 'rooftop skyline night', 'wet street night reflections'], places: ['New York', 'Atlanta', 'Los Angeles', 'Chicago'] },
  boombap: { ko: '붐뱁 · 힙합', want: '뉴욕의 어두운 밤 골목 · 브루클린 거리 · 비 오는 밤 지하철 입구 · 거친 벽돌 건물',
    prompts: ['a dark new york city alley at night with brick walls and fire escapes', 'a brooklyn street at night under orange street lights', 'a gritty urban street corner at night in new york', 'a rainy night in new york with steam and street lights'],
    terms: ['New York alley', 'Brooklyn street night', 'Harlem street', 'Bronx street', 'Manhattan night street'], shots: ['new york alley night', 'brooklyn street night', 'new york fire escape brick', 'nyc subway entrance night'], places: ['New York', 'Brooklyn', 'Harlem', 'Queens'] },
  lofi: { ko: '로파이', want: '비 오는 저녁의 작은 골목 · 따뜻한 불빛의 카페 · 도쿄 뒷골목 · 창가 너머 노을 · 조용한 서점',
    prompts: ['a quiet narrow street in the evening with warm cafe lights', 'a cozy cafe interior with warm lamps and plants', 'a rainy evening backstreet in tokyo with soft lights', 'a peaceful view of a city at golden hour from a window'],
    terms: ['cafe street evening', 'Tokyo backstreet', 'rainy evening street', 'cozy cafe interior', 'bookshop'], shots: ['cozy cafe window rain', 'tokyo backstreet evening', 'warm lights small street evening', 'bookstore cozy'], places: ['Tokyo', 'Kyoto', 'Seoul', 'Taipei', 'Paris'] },
  house: { ko: '하우스', want: '노을의 루프톱 바 · 해변 클럽 · 화려한 도시의 밤 · 빛나는 수영장 파티',
    prompts: ['a stylish rooftop bar at sunset overlooking a city', 'a beach club at sunset with colorful lights', 'a vibrant city at night full of lights', 'a glowing nightlife district with colorful neon'],
    terms: ['rooftop bar', 'beach club', 'nightlife district', 'Ibiza', 'promenade night'], shots: ['rooftop bar sunset city', 'beach club sunset lights', 'nightlife neon street'], places: ['Ibiza', 'Barcelona', 'Miami', 'London'] },
  techno: { ko: '테크노', want: '레이저와 조명이 가득한 클럽 · 불빛 흐르는 터널 · 미래적인 밤의 도시 · 베를린의 밤',
    prompts: ['a dark club with lasers and colorful stage lights', 'a futuristic tunnel with streaking lights', 'a futuristic city at night with blue and purple lights', 'a modern illuminated architecture at night'],
    terms: ['club lights', 'light tunnel', 'Berlin night', 'futuristic architecture night', 'illuminated bridge night'], shots: ['club lasers lights', 'light trails tunnel', 'futuristic city night lights'], places: ['Berlin', 'Tokyo', 'Seoul', 'Detroit'] },
  dnb: { ko: '드럼 앤 베이스', want: '빛줄기가 흐르는 밤의 고속도로 · 비에 젖은 네온 도시 · 미래적인 고층 빌딩 숲',
    prompts: ['a highway at night with long light trails', 'a rain soaked neon city at night', 'a futuristic skyline with glowing skyscrapers at night', 'a fast moving city at night with motion and lights'],
    terms: ['highway night light trails', 'neon city rain', 'skyline night', 'Hong Kong night', 'Shinjuku night'], shots: ['light trails highway night', 'neon city rain night', 'skyscrapers night glowing'], places: ['Hong Kong', 'Tokyo', 'London', 'Shanghai'] },
  kpop: { ko: 'K팝', want: '서울의 화려한 밤거리 · 한강 야경 · 감각적인 광장 · 반짝이는 쇼핑 거리',
    prompts: ['a colorful seoul street at night with bright signs', 'a city river with bridges lit up at night', 'a stylish modern plaza with colorful lights', 'a lively shopping street at night with sparkling lights'],
    terms: ['Seoul night', 'Myeongdong', 'Hongdae', 'Han River night', 'Gangnam street'], shots: ['seoul street night lights', 'han river night bridge', 'myeongdong night'], places: ['Seoul', 'Busan', 'Tokyo'] },
  ballad: { ko: '발라드', want: '노을 지는 호숫가 · 새벽 바닷가 · 가을 공원 길 · 비 오는 밤 창밖의 도시 불빛 · 꽃밭',
    prompts: ['a calm lake at golden hour with soft warm light', 'a quiet beach at sunrise with gentle waves', 'an autumn park path covered in golden leaves', 'city lights at night seen through a rainy window', 'a field of flowers at sunset'],
    terms: ['lake sunset', 'beach sunrise', 'autumn park', 'flower field', 'riverside sunset'], shots: ['lake golden hour', 'autumn leaves path', 'rainy window city lights', 'flower field sunset'], places: ['Jeju', 'Lake Bled', 'Hallstatt', 'Kyoto'] },
  rock: { ko: '록', want: '산을 가로지르는 시원한 도로 · 공연장 조명 · 해 질 녘 거친 해안 절벽 · 오래된 도시의 밤거리',
    prompts: ['an open road through dramatic mountains', 'a concert stage with bright spotlights', 'dramatic sea cliffs at sunset', 'a lively old city street at night with bars and lights'],
    terms: ['mountain road', 'concert stage', 'sea cliffs', 'Route 66', 'old town night bars'], shots: ['mountain road scenic', 'concert stage lights crowd', 'sea cliffs sunset'], places: ['Iceland', 'Scotland', 'California', 'London'] },
  reggaeton: { ko: '레게톤', want: '노을의 열대 해변 · 알록달록한 카리브 거리 · 야자수 · 따뜻한 밤의 광장',
    prompts: ['a tropical beach at sunset with palm trees', 'a colorful caribbean street with painted houses', 'a warm summer night in a lively latin plaza', 'a seaside promenade with palm trees at dusk'],
    terms: ['tropical beach', 'San Juan', 'Havana street', 'Cartagena', 'palm beach sunset'], shots: ['tropical beach sunset palm', 'colorful caribbean street', 'havana street classic car'], places: ['Puerto Rico', 'Havana', 'Cartagena', 'Miami'] },
  disco: { ko: '디스코', want: '반짝이는 미러볼 무대 · 빛나는 밤의 극장가 · 레트로 네온 거리',
    prompts: ['a glittering dance floor with a disco ball', 'a theater district at night with bright marquee lights', 'a retro street with colorful neon signs at night'],
    terms: ['theater district night', 'neon signs street', 'Broadway night', 'Las Vegas'], shots: ['disco ball lights', 'broadway marquee night', 'retro neon signs'], places: ['New York', 'Las Vegas', 'Paris'] },
}
/* 사진 검색어 더하기 (2026-11-21: 작품성 기준이 엄격해 2000~3000장을 채우려면 후보가 많이 필요해요) — 모두 "넓게 담은 장면" 쪽 (가까이 찍은 사진 빼기) */
export const SHOTS_MORE = {
  carol: ['christmas market square night', 'old town christmas illumination', 'snowy village christmas evening', 'christmas street decorations evening', 'advent market stalls lights', 'christmas tree plaza night', 'winter old town snow night', 'christmas lights city avenue', 'festive town square winter', 'christmas illuminated street europe'],
  waltz: ['vienna ringstrasse evening', 'baroque palace gardens', 'historic european square twilight', 'danube riverside evening', 'grand old town architecture dusk', 'parisian boulevard evening', 'opera house night illuminated', 'historic bridge sunset europe', 'palace fountain evening', 'old town river reflection night'],
  citypop: ['city skyline sunset', 'urban street golden hour', 'coastal road sunset drive', 'tokyo skyline dusk', 'los angeles sunset street', 'miami beach sunset skyline', 'neon street twilight', 'harbor city dusk lights', 'highway city lights evening', 'classic car street sunset', 'tokyo night skyline from tower', 'shibuya crossing dusk', 'yokohama minato mirai night', 'osaka dotonbori night', 'seoul skyline sunset', 'santa monica pier sunset', 'pacific coast highway sunset', 'honolulu waikiki sunset'],
  trap: ['city skyline night lights', 'downtown night street lights', 'skyscrapers night reflection', 'rooftop view city night', 'neon reflections wet street', 'urban night long exposure', 'city bridge night lights', 'night city aerial view', 'parking garage rooftop city night', 'times square night'],
  boombap: ['new york street night', 'brooklyn brownstone street', 'manhattan avenue night', 'new york fire escapes', 'harlem street evening', 'nyc subway station', 'new york brick buildings street', 'queens street evening', 'new york rain night street', 'bronx street view', 'brooklyn bridge night', 'manhattan bridge from dumbo', 'new york water towers rooftop', 'williamsburg bridge night', 'new york yellow taxi night', 'new york graffiti wall street', 'manhattan skyline night from brooklyn', 'new york street long exposure night'],
  lofi: ['rainy street evening lights', 'quiet alley evening lanterns', 'cafe terrace evening', 'kyoto street evening', 'tokyo side street night', 'cozy street dusk', 'bookshop street evening', 'small town evening lights', 'canal houses evening', 'lantern lit street'],
  house: ['beach sunset party lights', 'city nightlife street', 'rooftop terrace sunset city', 'ibiza sunset', 'marina night lights', 'promenade evening lights', 'festival stage lights night', 'pool party sunset', 'barcelona night street', 'colorful night city'],
  techno: ['berlin night city', 'illuminated modern architecture night', 'light installation night', 'futuristic building night', 'tunnel lights', 'industrial architecture night lights', 'laser show night', 'city lights long exposure', 'modern bridge night lights', 'night skyline blue lights'],
  dnb: ['light trails city night', 'hong kong skyline night', 'shinjuku night street', 'highway interchange night', 'night traffic long exposure', 'neon city night street', 'shanghai skyline night', 'rainy neon city', 'motion blur city night', 'futuristic skyline night'],
  kpop: ['seoul skyline night', 'han river night', 'myeongdong street', 'gangnam night', 'seoul tower night', 'busan night skyline', 'colorful shopping street night', 'hongdae street', 'seoul street evening lights', 'dongdaemun night'],
  ballad: ['lake sunset reflection', 'beach sunrise waves', 'autumn path trees', 'mountain lake morning', 'flower field sunset', 'misty lake dawn', 'seaside sunset calm', 'cherry blossom path', 'river sunset quiet', 'snowy forest path'],
  rock: ['mountain road landscape', 'coastal cliffs sunset', 'open road desert highway sunset', 'scottish highlands road', 'iceland road landscape', 'concert crowd stage lights', 'route 66 road', 'dramatic mountain landscape', 'canyon road sunset', 'coastal highway cliffs'],
  reggaeton: ['tropical beach sunset palms', 'colorful colonial street', 'havana street', 'caribbean beach', 'san juan old town', 'cartagena street colorful', 'palm beach promenade', 'tropical sunset sea', 'latin plaza evening', 'colorful houses street'],
  disco: ['broadway theater lights', 'las vegas strip night', 'neon signs street night', 'retro neon city', 'theater marquee night', 'glittering city night', 'dance hall lights', 'paris night lights', 'vintage neon signs street', 'colorful lights night street', 'las vegas neon night long exposure', 'hong kong neon street night', 'times square night lights', 'piccadilly circus night lights', 'ferris wheel lights night', 'moulin rouge night', 'neon arcade night street', 'tokyo neon street night'],
}
/* 장르마다 중심 장면 × 어울리는 도시 · 지역 (검색어를 늘려 후보를 3배로) */
export const SCENES_X = {
  carol: [['christmas market', 'christmas lights'], ['Strasbourg', 'Nuremberg', 'Vienna', 'Prague', 'Cologne', 'Dresden', 'Colmar', 'Tallinn', 'Zagreb', 'Budapest', 'Munich', 'Salzburg', 'Copenhagen', 'Brussels', 'Basel', 'Krakow']],
  waltz: [['old town evening', 'palace'], ['Vienna', 'Salzburg', 'Budapest', 'Prague', 'Paris', 'St Petersburg', 'Krakow', 'Ljubljana', 'Bruges', 'Lisbon', 'Seville', 'Florence']],
  citypop: [['skyline sunset', 'street dusk'], ['Tokyo', 'Osaka', 'Yokohama', 'Los Angeles', 'Miami', 'Honolulu', 'Hong Kong', 'Seoul', 'Busan', 'San Francisco', 'Singapore', 'Taipei']],
  trap: [['skyline night', 'downtown night'], ['Atlanta', 'New York', 'Chicago', 'Los Angeles', 'Houston', 'Toronto', 'Dubai', 'London', 'Detroit', 'Dallas', 'Philadelphia', 'Las Vegas']],
  boombap: [['street night', 'street'], ['Brooklyn', 'Manhattan', 'Harlem', 'Bronx', 'Queens', 'Greenwich Village', 'SoHo', 'Lower East Side', 'Williamsburg', 'Chinatown New York', 'Times Square', 'Bushwick']],
  lofi: [['street evening', 'alley night'], ['Kyoto', 'Tokyo', 'Taipei', 'Seoul', 'Osaka', 'Hanoi', 'Hoi An', 'Paris', 'Amsterdam', 'Lisbon', 'Kanazawa', 'Jiufen']],
  house: [['beach sunset', 'nightlife'], ['Ibiza', 'Barcelona', 'Miami', 'Mykonos', 'Tel Aviv', 'Rio de Janeiro', 'Bali', 'Cancun', 'Nice', 'Valencia', 'Lisbon', 'Amsterdam']],
  techno: [['night', 'illuminated architecture'], ['Berlin', 'Frankfurt', 'Detroit', 'Tokyo', 'Seoul', 'Shanghai', 'Rotterdam', 'Tbilisi', 'Leipzig', 'Hamburg', 'Oslo', 'Copenhagen']],
  dnb: [['night skyline', 'light trails'], ['Hong Kong', 'Shanghai', 'Shenzhen', 'Tokyo', 'London', 'Singapore', 'Bangkok', 'Kuala Lumpur', 'Chongqing', 'Dubai', 'Seoul', 'Taipei']],
  kpop: [['night', 'street night'], ['Seoul', 'Busan', 'Incheon', 'Gangnam', 'Myeongdong', 'Hongdae', 'Jongno', 'Itaewon', 'Jamsil', 'Yeouido', 'Daegu', 'Jeju']],
  ballad: [['lake sunset', 'landscape'], ['Jeju', 'Lake Bled', 'Hallstatt', 'Kyoto', 'Banff', 'Lake Como', 'Dolomites', 'Norway fjord', 'Scotland', 'New Zealand', 'Hokkaido', 'Swiss Alps']],
  rock: [['road landscape', 'cliffs'], ['Iceland', 'Scotland', 'California', 'Arizona', 'Utah', 'Norway', 'Ireland', 'Patagonia', 'Nevada', 'Cornwall', 'Faroe Islands', 'Big Sur']],
  reggaeton: [['beach', 'colorful street'], ['Puerto Rico', 'Havana', 'Cartagena', 'Medellin', 'Santo Domingo', 'Panama City', 'Cancun', 'Tulum', 'Punta Cana', 'Miami', 'Barbados', 'Aruba']],
  disco: [['night lights', 'neon'], ['Las Vegas', 'Broadway', 'Paris', 'Macau', 'Tokyo', 'London West End', 'Montreal', 'Melbourne', 'Shanghai', 'Chicago', 'Reno', 'Atlantic City']],
}
export const shotTerms = (g, places = 4) => { const b = BRIEFS[g]; if (!b) return []; const [sc, pl] = SCENES_X[g] || [[], []], x = []; for (const p of pl) for (const c of sc) x.push(`${p} ${c}`); return [...new Set([...b.shots, ...(SHOTS_MORE[g] || []), ...b.places.slice(0, places).map((pl) => `${b.shots[0].split(' ').slice(0, 2).join(' ')} ${pl}`), ...x])] }
export const GENRES = Object.keys(BRIEFS)

/* 감성: 사람에게 멋짐 · 낭만 · 추억 같은 좋은 감정을 불러일으키는 사진 */
export const APPEAL = ['a beautiful scenic travel photo that makes you want to visit', 'a breathtaking romantic view', 'a cool cinematic scene', 'a cozy warm atmospheric place', 'a nostalgic golden hour scene', 'a magical night with glowing lights']
/* 평범 · 지루 (감성 점수의 기준선): 대시캠 도로 · 아무 데나 찍은 길 · 텅 빈 방 */
export const PLAIN = ['an ordinary boring photo of a road', 'a random street photo taken from a car dashcam', 'a plain empty room', 'a featureless field', 'an empty parking lot']
/* 불쾌 · 이상한 느낌 — 하나라도 강하면 빼요 (사용자 지적: 사막 · 생선 뼈 박물관 · 공장의 이상한 그래피티 · 묘지 · 병원 …) */
export const REJECT = [
  ['묘지', 'a cemetery with gravestones'], ['전쟁 기념', 'a war memorial or military cemetery'], ['병원', 'a hospital room or clinic'], ['뼈 · 표본', 'skeletons, bones or animal specimens in a museum'],
  ['죽은 동물', 'a dead animal'], ['폐허', 'an abandoned derelict ruined building'], ['낡은 공장', 'a dirty industrial factory or warehouse interior'], ['낙서 폐공장', 'graffiti on an abandoned factory wall'],
  ['쓰레기', 'trash, garbage and litter'], ['공사장', 'a construction site'], ['화장실', 'a bathroom or toilet'], ['황무지 사막', 'a barren empty desert wasteland'],
  ['감옥', 'a prison or jail'], ['성당 안 · 제단', 'a church interior with an altar'], ['사무실', 'an office with desks and cubicles'], ['빈 촬영 스튜디오', 'an empty white photo studio'],
  ['어둡고 흐림', 'a very dark blurry low quality photo'], ['좁은 복도', 'a narrow empty corridor'], ['주유소 · 차고', 'a gas station or garage'], ['동굴 · 지하', 'a dark cave or underground tunnel'],
  ['군사', 'military vehicles or weapons'], ['으스스함', 'a creepy scary place'], ['사람 얼굴 가까이', 'a close up of people faces'], ['주차장', 'a multi storey car park'],
  /* 1차 검토 시트(2026-11-21)에서 찾은 것: 렌즈에 맺힌 물방울 · 찍는 사람 뒤통수 · 아이 · 우주에서 본 지도 · 텅 빈 연회장 */
  ['그림 · 옛 지도', 'a painting, drawing or illustrated old map'], ['셀카 · 포즈', 'people posing for a selfie'], ['글자 얹은 카드', 'a greeting card or poster with text overlay'], ['옛 흑백 사진', 'an old sepia or black and white historical photo'], /* 3차 시트: 새해 카드 · 세피아 옛 사진 */ /* 2차 시트(2026-11-21 작품성): 옛 지도 그림 · 셀카 */
  ['렌즈 물방울', 'a blurry photo with water drops on the camera lens'], ['찍는 사람 머리', 'the back of a person head blocking the camera view'], ['어린이', 'a young child or little kids'], ['위성 지도', 'a satellite map of the earth from space'], ['빈 연회장', 'an empty banquet hall'],
]
export const ALL_PROMPTS = () => { const out = []; for (const [g, b] of Object.entries(BRIEFS)) for (const p of b.prompts) out.push(['g:' + g, p]); for (const p of APPEAL) out.push(['appeal', p]); for (const p of PLAIN) out.push(['plain', p]); for (const [k, p] of REJECT) out.push(['rej:' + k, p]); out.push(...CRAFT_PROMPTS()); return out } /* 작품성 문장은 맨 뒤 — 감성 · 거절 · 장르 softmax 에는 안 들어가요 */

/* 점수: v = 이미지 임베딩(정규화) · T = ALL_PROMPTS 순서의 문장 임베딩(정규화). CLIP 의 로짓 배율 100 으로 softmax.
   감성 = P(감성) / (P(감성) + P(평범))   · 거절 = 가장 강한 거절 개념의 확률(전체 softmax 안에서) · 장르 = 장르마다 자기 문장들 중 최댓값의 softmax(장르끼리) × 장르 쪽 총확률 */
export function scoreScene(v, T, prompts = ALL_PROMPTS(), tech = null) {
  const lgAll = T.map((t) => { let s = 0; for (let k = 0; k < t.length; k++) s += t[k] * v[k]; return 100 * s }), keysAll = prompts.map((p) => p[0]), main = prompts.map((p, i) => [p, i]).filter(([p]) => !isCraftKey(p[0])), mainP = main.map(([p]) => p), lg = main.map(([, i]) => lgAll[i])
  const craft = { pro: +axis(lgAll, keysAll, 'pro', 'am').toFixed(3), close: +axis(lgAll, keysAll, 'close', 'wide').toFixed(3), aes: aesScore(v), ...(tech ? { tech } : {}) } /* 전문 vs 아마추어 · 가까이 vs 넓게 · 미학 점수 */
  prompts = mainP; const mx = Math.max(...lg), ex = lg.map((x) => Math.exp(x - mx)), Z = ex.reduce((a, b) => a + b, 0), P = ex.map((x) => x / Z)
  let pa = 0, pp = 0, pg = 0, rej = 0, rejK = ''; const gmax = {}
  prompts.forEach(([k], i) => { if (k === 'appeal') pa += P[i]; else if (k === 'plain') pp += P[i]; else if (k.startsWith('rej:')) { if (P[i] > rej) { rej = P[i]; rejK = k.slice(4) } } else { pg += P[i]; const g = k.slice(2); gmax[g] = Math.max(gmax[g] ?? -1e9, lg[i]) } })
  const gk = Object.keys(gmax), gm = Math.max(...gk.map((g) => gmax[g])), ge = gk.map((g) => Math.exp((gmax[g] - gm) / 2)), gz = ge.reduce((a, b) => a + b, 0) /* 장르끼리는 부드럽게(온도 2) — 한 장면이 여러 장르에 어울릴 수 있어요 */
  const fit = {}; gk.forEach((g, i) => { fit[g] = +(ge[i] / gz).toFixed(3) })
  return { appeal: +(pa / Math.max(1e-9, pa + pp)).toFixed(3), plain: +pp.toFixed(3), rej: +rej.toFixed(3), rejK, genreMass: +pg.toFixed(3), fit, best: gk.reduce((a, g) => (fit[g] > (fit[a] ?? -1) ? g : a), gk[0]), ...craft }
}
/* 판정 (기본값은 Actions 의 검토 시트로 확인해 고쳐요): 거절 개념이 강하면 빼고, 감성이 낮거나 장르 쪽 확률이 너무 낮으면(어느 노래에도 안 어울림) 빼요 */
/* 장르별 미학 하한 (2026-11-22 사용자: "평균 미감 점수가 높은 장르일수록 평균이 더 높았으면") — 하한 6.25 + 1.5 × max(0, 수축 평균 − 전체 평균),
   수축 평균 = (n · 장르 평균 + 30 · 전체 평균) / (n + 30) (사진이 적은 장르가 우연히 튀지 않게). 2224장 기준 전체 평균 6.88 에서 계산 — 노을 · 야경이 많은 장르는 더 엄격하게 */
export const AES_G = { carol: 6.53, reggaeton: 6.45, boombap: 6.34, ballad: 6.34, rock: 6.29, kpop: 6.26 }
export const CUT = { rej: 0.12, appeal: 0.55, genreMass: 0.25, close: 0.2, pro: 0.55, aes: 6.25, aesG: AES_G, tech: TECH_CUT } /* 2차 시트: 미학 5.0~5.3 에 셀카 · 군중 얼굴 · 평범한 거리 · 위성 지도 → 5.5, 가까이 0.22~0.40 에 음식 노점 · 물건 → 0.25, 3차 시트 0.20~0.25 에 가게 앞 · 장식 앞 · 캐릭터 진열 → 0.20, 미학 5.5 → 6.0 (사용자: "잘 찍은 사진" 이상만, 2026-11-22) */ /* close · pro · aes: 작품성 (검토 시트로 정해요 · 옛 채점처럼 값이 없으면 그 검사는 건너뛰어요) */
export function verdict(sc, cut = CUT) {
  if (sc.rej >= cut.rej) return { ok: false, why: '불쾌 · 이상: ' + sc.rejK }
  if (sc.appeal < cut.appeal && !(cut.gmAlt != null && sc.genreMass >= cut.gmAlt)) return { ok: false, why: '평범 · 지루함' } /* gmAlt: 거리에서 찍은 사진이라 "평범" 쪽으로 기울어도 장르 분위기가 아주 강하면(밤 쇼핑 거리 · 네온) 살려요 */
  if (sc.genreMass < cut.genreMass) return { ok: false, why: '어느 장르 분위기와도 멀어요' }
  if (sc.close != null && cut.close != null && sc.close > cut.close) return { ok: false, why: '가까이 찍었어요 (클로즈업)' }
  if (sc.pro != null && cut.pro != null && sc.pro < cut.pro) return { ok: false, why: '작품성: 스냅 사진 같아요' }
  if (sc.aes != null && cut.aes != null && sc.aes < Math.max(cut.aes, (cut.aesG && cut.aesG[sc.best]) || 0)) return { ok: false, why: '작품성: 미학 점수가 낮아요' } /* 장르별 하한(aesG)은 가장 맞는 장르 기준 */
  if (sc.tech) { const t = techVerdict(sc.tech, cut.tech || TECH_CUT); if (!t.ok) return { ok: false, why: '작품성 · ' + t.why } }
  return { ok: true, why: '' }
}
/* 장면 목록에 싣는 짧은 장르 적합도: 상위 3개만 [장르, 0–255] */
export const packFit = (fit) => Object.entries(fit).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([g, x]) => [g, Math.round(x * 255)])
export const unpackFit = (g) => Object.fromEntries((g || []).map(([k, x]) => [k, x / 255]))
