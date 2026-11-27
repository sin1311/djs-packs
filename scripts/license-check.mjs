// 모델 라이선스 원문 확인(사용자 지시: 원문 확인 뒤에만 넣기) — Hugging Face 모델 카드 · 메타데이터 · LICENSE 파일 + GitHub LICENSE 를 그대로 받아 licenses/<묶음>/ 에 남겨요.
//  해석하지 않아요: 원문 · 메타데이터의 license 칸 · 파일 목록(ONNX 판이 있는지)만. 판단은 사람이 원문을 읽고.
import { mkdirSync, writeFileSync } from 'node:fs'
const group = process.argv[2] || 'vision', hf = (process.argv[3] || '').split(',').filter(Boolean), gh = (process.argv[4] || '').split(',').filter(Boolean)
const out = `licenses/${group}`; mkdirSync(out, { recursive: true })
const get = async (u, as = 'text') => { for (let i = 0; i < 3; i++) { try { const r = await fetch(u, { headers: { 'user-agent': 'djs-packs-license-check' } }); if (r.status === 404) return null; if (r.ok) return as === 'json' ? r.json() : r.text() } catch (e) { /* 다시 */ } await new Promise((s) => setTimeout(s, 1500 * (i + 1))) } return undefined }
const safe = (s) => s.replace(/[^A-Za-z0-9._-]+/g, '__'), rows = []
for (const id of hf) {
  const meta = await get(`https://huggingface.co/api/models/${id}`, 'json'), dir = `${out}/${safe(id)}`; mkdirSync(dir, { recursive: true })
  if (!meta) { rows.push({ id, src: 'hf', ok: false, why: meta === null ? '없음(404)' : '못 받음' }); continue }
  const files = (meta.siblings || []).map((x) => x.rfilename), lic = (meta.cardData && meta.cardData.license) || null, licTags = (meta.tags || []).filter((t) => t.startsWith('license:')), licName = (meta.cardData && (meta.cardData.license_name || meta.cardData.license_link)) || null
  writeFileSync(`${dir}/meta.json`, JSON.stringify({ id, license: lic, licenseTags: licTags, licenseName: licName, gated: meta.gated || false, pipeline: meta.pipeline_tag || null, params: meta.safetensors && meta.safetensors.total || null, datasets: (meta.cardData && meta.cardData.datasets) || null, onnx: files.filter((f) => /\.onnx$/.test(f)), files: files.slice(0, 200), sha: meta.sha, lastModified: meta.lastModified }, null, 2))
  const card = await get(`https://huggingface.co/${id}/raw/main/README.md`); if (card) writeFileSync(`${dir}/README.md`, card)
  const lf = files.filter((f) => /^(LICENSE|LICENCE|license|licence)(\.[a-z]+)?$|LICENSE/i.test(f)).slice(0, 4); const got = []
  for (const f of lf) { const t = await get(`https://huggingface.co/${id}/raw/main/${f}`); if (t) { writeFileSync(`${dir}/${safe(f)}`, t); got.push(f) } }
  rows.push({ id, src: 'hf', ok: true, license: lic, tags: licTags.join(' '), licenseFiles: got, onnx: files.filter((f) => /\.onnx$/.test(f)).length, gated: !!meta.gated, params: meta.safetensors && meta.safetensors.total || null })
}
for (const spec of gh) { /* owner/repo@ref:path */
  const [repo, rest = 'HEAD:LICENSE'] = spec.split('@'), [ref, path] = rest.split(':'), t = await get(`https://raw.githubusercontent.com/${repo}/${ref}/${path}`), dir = `${out}/${safe(repo)}`; mkdirSync(dir, { recursive: true })
  if (t) writeFileSync(`${dir}/${safe(path)}`, t); rows.push({ id: repo, src: 'github', ok: !!t, path, firstLine: t ? t.split('\n').find((l) => l.trim()).trim().slice(0, 120) : null })
}
writeFileSync(`${out}/summary.json`, JSON.stringify({ at: new Date().toISOString(), rows }, null, 2))
writeFileSync(`${out}/SUMMARY.md`, `# 라이선스 원문 모음 — ${group} (${new Date().toISOString().slice(0, 10)})\n\n원문은 각 폴더(README.md = 모델 카드 · LICENSE* · meta.json). 이 표는 메타데이터를 옮겨 적은 것일 뿐 법적 판단이 아니에요.\n\n| 모델 · 저장소 | 출처 | 받음 | license 칸 | 태그 | LICENSE 파일 | ONNX 파일 수 | 게이트 | 매개변수 |\n|---|---|---|---|---|---|---|---|---|\n` + rows.map((r) => `| ${r.id} | ${r.src} | ${r.ok ? '예' : '아니요 ' + (r.why || '')} | ${r.license || r.firstLine || ''} | ${r.tags || ''} | ${(r.licenseFiles || []).join(' ') || (r.path || '')} | ${r.onnx ?? ''} | ${r.gated ? '예' : ''} | ${r.params || ''} |`).join('\n') + '\n')
console.log(JSON.stringify(rows, null, 1))
