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
