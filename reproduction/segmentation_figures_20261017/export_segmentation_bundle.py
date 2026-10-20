#!/usr/bin/env python3
"""Export alignment JSONL files to multi-tier TextGrid and interactive HTML."""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import wave
from collections import defaultdict
from pathlib import Path


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alignment",
        action="append",
        required=True,
        metavar="LABEL=ALIGNMENTS_JSONL",
        help="Repeat for each model and split; summary.json must be beside JSONL.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples-per-split", type=int, default=4)
    parser.add_argument("--title", default="Speech segmentation comparison")
    return parser.parse_args()


def quoted(value: str) -> str:
    return '"' + value.replace('"', '""').replace("\n", " ") + '"'


def interval_tier(name: str, labels: list[str], starts: list[float], ends: list[float], xmax: float) -> list[str]:
    items: list[tuple[float, float, str]] = []
    cursor = 0.0
    for label, start, end in zip(labels, starts, ends):
        start = max(cursor, min(float(start), xmax))
        end = max(start, min(float(end), xmax))
        if start > cursor + 1e-8:
            items.append((cursor, start, ""))
        if end > start + 1e-8:
            items.append((start, end, str(label)))
        cursor = max(cursor, end)
    if cursor < xmax - 1e-8:
        items.append((cursor, xmax, ""))
    lines = [
        '        class = "IntervalTier"',
        f"        name = {quoted(name)}",
        "        xmin = 0",
        f"        xmax = {xmax:.9f}",
        f"        intervals: size = {len(items)}",
    ]
    for i, (start, end, label) in enumerate(items, 1):
        lines.extend(
            [
                f"        intervals [{i}]:",
                f"            xmin = {start:.9f}",
                f"            xmax = {end:.9f}",
                f"            text = {quoted(label)}",
            ]
        )
    return lines


def point_tier(name: str, points: list[float], xmax: float) -> list[str]:
    points = sorted({max(0.0, min(float(x), xmax)) for x in points})
    lines = [
        '        class = "TextTier"',
        f"        name = {quoted(name)}",
        "        xmin = 0",
        f"        xmax = {xmax:.9f}",
        f"        points: size = {len(points)}",
    ]
    for i, point in enumerate(points, 1):
        lines.extend(
            [
                f"        points [{i}]:",
                f"            number = {point:.9f}",
                '            mark = "boundary"',
            ]
        )
    return lines


def write_textgrid(path: Path, rows: dict[str, dict]) -> None:
    first = next(iter(rows.values()))
    xmax = float(first["duration"])
    tiers: list[list[str]] = []
    tiers.append(interval_tier("reference_ARPAbet", first["gt_phones"], first["gt_starts"], first["gt_ends"], xmax))
    tiers.append(point_tier("reference_boundaries", first.get("gt_boundaries", first["gt_ends"][:-1]), xmax))
    for label, row in rows.items():
        safe = label.replace(" ", "_")
        tiers.append(interval_tier(f"{safe}_ARPAbet", row["phones"], row["pred_starts"], row["pred_ends"], xmax))
        tiers.append(point_tier(f"{safe}_boundaries", row.get("pred_boundaries", row["pred_ends"][:-1]), xmax))
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {xmax:.9f}",
        "tiers? <exists>",
        f"size = {len(tiers)}",
        "item []:",
    ]
    for i, tier in enumerate(tiers, 1):
        lines.append(f"    item [{i}]:")
        lines.extend(tier)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def waveform_peaks(path: Path, bins: int = 700) -> list[float]:
    with wave.open(str(path), "rb") as stream:
        width = stream.getsampwidth()
        channels = stream.getnchannels()
        frames = stream.getnframes()
        raw = stream.readframes(frames)
    if width != 2:
        return [0.0] * bins
    count = len(raw) // 2
    samples = [int.from_bytes(raw[i : i + 2], "little", signed=True) for i in range(0, len(raw), 2)]
    if channels > 1:
        samples = samples[::channels]
    step = max(1, math.ceil(len(samples) / bins))
    peaks = [max((abs(x) for x in samples[i : i + step]), default=0) / 32768.0 for i in range(0, len(samples), step)]
    return [round(x, 4) for x in peaks[:bins]]


def choose_examples(records: dict[str, dict[str, dict]], count: int) -> list[str]:
    candidates = []
    for recording_id, models in records.items():
        row = next(iter(models.values()))
        duration = float(row["duration"])
        if not 2.5 <= duration <= 9.0:
            continue
        scores = [float(x["metrics"]["boundary_only_20ms_f1_pct"]) for x in models.values()]
        candidates.append((sum(scores) / len(scores), max(scores) - min(scores), recording_id))
    if not candidates:
        return sorted(records)[:count]
    candidates.sort()
    picks: list[str] = []
    if len(candidates) > 1:
        for q in (0.1, 0.5, 0.9):
            picks.append(candidates[round(q * (len(candidates) - 1))][2])
    picks.extend(x[2] for x in sorted(candidates, key=lambda x: x[1], reverse=True))
    for _, _, recording_id in candidates:
        if recording_id not in picks:
            picks.append(recording_id)
    return list(dict.fromkeys(picks))[:count]


def build_html(title: str, data: dict, output: Path) -> None:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--ink:#17202a;--muted:#64748b;--paper:#f8fafc;--line:#cbd5e1;--ref:#2563eb;--m1:#d97706;--m2:#059669;--play:#dc2626}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.35 ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1400px;margin:auto;padding:18px}} h1{{font-size:22px;margin:0 0 12px}} .controls{{display:flex;gap:12px;align-items:end;flex-wrap:wrap;margin-bottom:12px}}
label{{display:grid;gap:4px;color:var(--muted);font-size:12px}} select,input{{font:inherit;padding:7px;border:1px solid var(--line);background:white;border-radius:5px}}
audio{{width:min(620px,100%);height:36px}} .metrics{{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0 12px;color:var(--muted)}} .metrics b{{color:var(--ink)}}
#chart{{position:relative;border:1px solid var(--line);background:white;overflow:hidden;min-height:360px}} svg{{display:block;width:100%}} .phone{{cursor:pointer}} .phone text{{pointer-events:none;font-size:11px;fill:var(--ink)}}
.axis{{fill:var(--muted);font-size:11px}} .lane-label{{font-weight:650;font-size:12px;fill:var(--ink)}} .tip{{position:absolute;display:none;pointer-events:none;background:#111827;color:white;padding:6px 8px;border-radius:4px;font-size:12px;z-index:2}}
.legend{{display:flex;gap:15px;flex-wrap:wrap;margin:10px 0;color:var(--muted)}} .swatch{{display:inline-block;width:11px;height:11px;margin-right:5px;vertical-align:-1px}}
@media(max-width:600px){{main{{padding:10px}} h1{{font-size:18px}} .controls label{{width:100%}} select{{width:100%}}}}
</style></head><body><main><h1>{html.escape(title)}</h1>
<div class="controls"><label>Split<select id="split"></select></label><label>Utterance<select id="utt"></select></label><label>Window <input id="zoom" type="range" min="1" max="5" step="0.25" value="1"></label><audio id="audio" controls preload="metadata"></audio></div>
<div class="metrics" id="metrics"></div><div class="legend" id="legend"></div><div id="chart"><div class="tip" id="tip"></div><svg id="svg" role="img" aria-label="Reference and model phone segmentation aligned to audio"></svg></div>
<script id="seg-data" type="application/json">{payload}</script>
<script>
const D=JSON.parse(document.getElementById('seg-data').textContent), split=document.getElementById('split'), utt=document.getElementById('utt'), audio=document.getElementById('audio'), svg=document.getElementById('svg'), tip=document.getElementById('tip'), zoom=document.getElementById('zoom');
const colors=['var(--m1)','var(--m2)','#7c3aed','#db2777']; let current=null;
for(const s of Object.keys(D.splits)) split.add(new Option(s,s));
function fillUtts(){{utt.innerHTML=''; for(const id of Object.keys(D.splits[split.value])) utt.add(new Option(id,id)); render();}}
function el(name,attrs={{}}){{const n=document.createElementNS('http://www.w3.org/2000/svg',name); for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v); return n}}
function render(){{if(!utt.value)return; current=D.splits[split.value][utt.value]; audio.src=current.audio; const dur=current.duration,z=+zoom.value, center=Math.min(dur,Math.max(0,audio.currentTime||dur/2)), span=dur/z, lo=Math.max(0,Math.min(dur-span,center-span/2)), hi=lo+span; const W=Math.max(320,document.getElementById('chart').clientWidth), left=92,right=12,plot=W-left-right,laneH=54,waveH=85,models=Object.keys(current.models),H=waveH+(models.length+1)*laneH+34; svg.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`); svg.innerHTML=''; const x=t=>left+(t-lo)/(hi-lo)*plot;
const peaks=current.peaks, mid=waveH/2+10,path=[]; peaks.forEach((p,i)=>{{const t=i/(peaks.length-1)*dur;if(t<lo||t>hi)return;const xx=x(t),h=p*32;path.push(`M${{xx.toFixed(1)}},${{(mid-h).toFixed(1)}}L${{xx.toFixed(1)}},${{(mid+h).toFixed(1)}}`)}}); svg.appendChild(el('path',{{d:path.join(''),'stroke':'var(--muted)','stroke-width':'1','opacity':'.6'}}));
const lanes=[['Reference',current.reference,'var(--ref)'],...models.map((m,i)=>[m,current.models[m],colors[i%colors.length]])]; lanes.forEach(([name,row,color],li)=>{{const y=waveH+li*laneH; const label=el('text',{{x:8,y:y+30,class:'lane-label'}});label.textContent=name;svg.appendChild(label); row.phones.forEach((ph,i)=>{{const a=row.starts[i],b=row.ends[i];if(b<lo||a>hi)return;const g=el('g',{{class:'phone'}}),r=el('rect',{{x:x(Math.max(a,lo)),y:y+7,width:Math.max(1,x(Math.min(b,hi))-x(Math.max(a,lo))),height:34,fill:color,'fill-opacity':li?'.18':'.14',stroke:color,'stroke-width':'1'}}),tx=el('text',{{x:x(Math.max(a,lo))+3,y:y+29}});tx.textContent=ph;g.append(r,tx);g.addEventListener('click',()=>{{audio.currentTime=a;audio.play()}});g.addEventListener('mousemove',e=>{{tip.style.display='block';tip.style.left=(e.offsetX+12)+'px';tip.style.top=(e.offsetY+8)+'px';tip.textContent=`${{name}} · ${{ph}} · ${{a.toFixed(3)}}–${{b.toFixed(3)}} s`}});g.addEventListener('mouseleave',()=>tip.style.display='none');svg.appendChild(g)}})}});
for(let t=Math.ceil(lo);t<=hi;t++){{const xx=x(t),line=el('line',{{x1:xx,y1:waveH-8,x2:xx,y2:H-25,stroke:'var(--line)','stroke-dasharray':'2 3'}}),text=el('text',{{x:xx+2,y:H-8,class:'axis'}});text.textContent=t+'s';svg.append(line,text)}} const play=el('line',{{id:'playhead',x1:x(audio.currentTime),x2:x(audio.currentTime),y1:0,y2:H-25,stroke:'var(--play)','stroke-width':'2'}});svg.appendChild(play);
document.getElementById('metrics').innerHTML=models.map(m=>`<span><b>${{m}}</b> F1@20 ${{current.models[m].f1.toFixed(2)}}% · P ${{current.models[m].precision.toFixed(2)}}% · R ${{current.models[m].recall.toFixed(2)}}%</span>`).join(''); document.getElementById('legend').innerHTML=`<span><i class="swatch" style="background:var(--ref)"></i>ARPAbet reference</span>`+models.map((m,i)=>`<span><i class="swatch" style="background:${{colors[i%colors.length]}}"></i>${{m}} ARPAbet</span>`).join('');}}
split.addEventListener('change',fillUtts);utt.addEventListener('change',render);zoom.addEventListener('input',render);audio.addEventListener('timeupdate',()=>{{const p=document.getElementById('playhead');if(p)render()}});window.addEventListener('resize',render);fillUtts();
</script></main></body></html>"""
    output.write_text(doc, encoding="utf-8")


def main() -> int:
    cfg = args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_raw = json.loads(cfg.manifest.read_text(encoding="utf-8"))
    manifest = {Path(str(x["wav"])).stem: x for x in manifest_raw.values()}
    by_split: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    summaries: dict[str, dict[str, dict]] = defaultdict(dict)
    for spec in cfg.alignment:
        if "=" not in spec:
            raise ValueError(f"Expected LABEL=PATH, got {spec!r}")
        label, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
        split = str(summary["split"])
        summaries[label][split] = summary
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                by_split[split][str(row["recording_id"])][label] = row

    textgrid_count = 0
    for split, records in by_split.items():
        for recording_id, rows in records.items():
            write_textgrid(cfg.output_dir / "textgrids" / split / f"{recording_id}.TextGrid", rows)
            textgrid_count += 1

    web: dict = {"splits": {}}
    selected: dict[str, list[str]] = {}
    audio_dir = cfg.output_dir / "interactive" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for split, records in by_split.items():
        chosen = choose_examples(records, cfg.examples_per_split)
        selected[split] = chosen
        web["splits"][split] = {}
        for recording_id in chosen:
            rows = records[recording_id]
            first = next(iter(rows.values()))
            wav = Path(str(manifest[recording_id]["wav"]))
            dst = audio_dir / f"{recording_id}{wav.suffix}"
            shutil.copy2(wav, dst)
            models = {}
            for label, row in rows.items():
                metrics = row["metrics"]
                models[label] = {
                    "phones": row["phones"], "starts": row["pred_starts"], "ends": row["pred_ends"],
                    "f1": metrics["boundary_only_20ms_f1_pct"],
                    "precision": metrics["boundary_only_20ms_precision_pct"],
                    "recall": metrics["boundary_only_20ms_recall_pct"],
                }
            web["splits"][split][recording_id] = {
                "duration": first["duration"], "audio": f"audio/{dst.name}", "peaks": waveform_peaks(wav),
                "reference": {"phones": first["gt_phones"], "starts": first["gt_starts"], "ends": first["gt_ends"]},
                "models": models,
            }
    interactive = cfg.output_dir / "interactive"
    build_html(cfg.title, web, interactive / "index.html")
    result = {
        "title": cfg.title,
        "manifest": str(cfg.manifest.resolve()),
        "textgrid_count": textgrid_count,
        "selected_examples": selected,
        "summaries": summaries,
        "interactive_html": str((interactive / "index.html").resolve()),
    }
    (cfg.output_dir / "bundle_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"textgrid_count": textgrid_count, "selected_examples": selected, "html": result["interactive_html"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
