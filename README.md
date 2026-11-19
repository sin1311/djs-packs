# djs-packs — AI 뮤직 DJ 스튜디오 음원 팩

이 저장소는 AI 뮤직 DJ 스튜디오가 GPU 등급에 따라 받아 가는 **실제 악기 샘플 팩**만 담아요(앱 코드는 따로 있어요).
악기 한 음씩 녹음한 공개 라이선스 샘플과, 이 프로젝트가 직접 합성한 소리뿐이에요. 저작권 곡은 들어 있지 않아요.

| 가지 | 대상 | 내용 |
|---|---|---|
| `lite` | T0 · T1 (GPU 없음 · 저사양) | 피아노 세기 4단 · 약 8.6MB (앱에도 들어 있어요) |
| `standard` | T2 · T3 | 피아노 세기 6단 · 더 긴 음 · 약 23MB |
| `full` | T4 · T5 | 피아노 세기 10단 · 가장 긴 음 · 약 45MB |
| `main` | — | 팩 공장(`scripts/build-packs.mjs`) · 워크플로 |

받는 주소: `https://cdn.jsdelivr.net/gh/sin1311/djs-packs@<가지>/manifest.json` (막히면 `https://raw.githubusercontent.com/sin1311/djs-packs/<가지>/`)

## 출처 · 라이선스
- **Salamander Grand Piano V3** — Alexander Holm, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). 원본: https://archive.org/details/SalamanderGrandPianoV3 (sfz 정리본: GitHub sfzinstruments/SalamanderGrandPiano). 변경: 세기 레이어 일부만 골라 길이를 줄이고 Opus 로 다시 압축했어요.
- **VS Chamber Orchestra: Community Edition** — Versilian Studios (Sam Gossner · Simon Dalzell 외), [CC0](https://creativecommons.org/publicdomain/zero/1.0/). https://github.com/sgossner/VSCO-2-CE — 현악 · 플루트 · 트럼펫 · 하프 · 콘트라베이스.
- **어쿠스틱 기타 · 공간 응답(IR) 3종** — 이 프로젝트가 물리 모델(확장 카플러스-스트롱 · 영상원 + 지수 감쇠)로 직접 합성, CC0.

각 팩의 `manifest.json` 에 파일마다 sha256 · 출처 · 라이선스가 들어 있고, 앱은 라이선스 검사(상업 배포 안전 모드)와 sha256 확인을 통과한 팩만 써요.

## 다시 만들기
Actions 탭 → **build-packs** → **Run workflow**. 로컬에서는 `node scripts/build-packs.mjs --tier standard --out out/standard` (ffmpeg · git · 인터넷 필요).


## 장면 팩 (v1.0-C) — `scenes` 가지
- 만들기: Actions → **build-scenes** → Run workflow. 주제 10개(노을 도시 · 도시 야경 · 한낮 도심 · 겨울 광장 · 클럽 · 노을 들판 · 숲 · 바다 · 공장 · 아늑한 실내) × count 개를 Poly Haven 태그로 자동으로 골라요. 장면은 `scenes-1` · `scenes-2` … 가지에 40MB 씩 나눠 담고, 목록 `index.json` 은 `scenes` 가지에 있어요. `scripts/scene-factory.mjs` (앱 저장소의 같은 파일 사본) 이 Poly Haven(CC0) HDRI 를 받아 깊이 · 지면 캘리브레이션 · 2~3층 LDI 팩으로 만들어요.
- 모델: Depth Anything V2 **Small** (Apache-2.0 — Base/Large 는 비상업이라 쓰지 않아요) · YuNet 얼굴 검출 (MIT). 번호판은 휴리스틱 후보를 흐리고 사람 확인 목록(`report.json`)에 올려요.
- 형식: `index.json` (djs-scenes/1) → 장면마다 `manifest.json` (djs-scene/1, 파일마다 sha256) · 층마다 색 `l<k>.webp`(알파 포함) · 거리 `l<k>-dist.png`(log 거리 16비트, R = 위 · G = 아래 8비트).
- 장면 카드에는 "재구성 장면 · CC 자산" 처럼 정직하게 표시돼요. 출처 · 저작자는 manifest 의 `sources` 에 있어요.
