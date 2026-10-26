"""Verify recorded artifacts and package the latest local posterior export."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

HERE = Path(__file__).resolve().parent


def main():
    output = Path(json.loads((HERE / 'LATEST_EXPORT.json').read_text())['output'])
    complete = json.loads((output / 'COMPLETE.json').read_text())
    for name, digest in complete['files'].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest, name
    manifest = json.loads((output / 'manifest.json').read_text())
    stats = json.loads((output / 'statistics.json').read_text())
    browser = dict(verified=False, reason='No successful browser selftest recorded; data and schema checks passed.')
    for path in [output / 'browser_validation.log', HERE / 'browser_retry.log', HERE / 'browser_unrestricted.log']:
        if not path.exists():
            continue
        match = re.search(r'POSTERIOR_SELFTEST (\{[^\n]+\})', path.read_text())
        if match:
            result = json.loads(match.group(1))
            assert result['pass'] and result['full42'] and result['overview_visible']
            browser = dict(result, verified=True, webgl_3d_rotation_verified=False,
                           tested_mode='Firefox headless heatmap fallback, four methods and full42', source_log=path.name)
            break
    (output / 'BROWSER_VALIDATION.json').write_text(json.dumps(browser, indent=2) + '\n')
    lines = ['# Buckeye posterior 导出结果', '',
             '已完成 seed43 四模型的本地 posterior 导出。使用 full40 dev 第一条 `buckeye_s0701a_0000`（okay one two，1.518875秒），每模型150帧 × 42类。', '',
             '**模型仍在训练；使用导出开始时的 single best-dev 快照，不能视为最终模型或全数据集统计。**', '',
             '| 模型 | epoch | step | 平均最大概率 | 平均熵 nats | top1-top2间隔 | argmax切换次数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for member in manifest['methods']:
        s = stats['methods'][member['cost']]
        lines.append(f"| {member['title']} | {member['epoch']} | {member['step']} | {s['mean_max_posterior']:.6f} | "
                     f"{s['mean_entropy_nats']:.6f} | {s['mean_top1_top2_margin']:.6f} | {s['argmax_switches']} |")
    lines += ['', '全42类原始 softmax；统计涵盖本句所有帧。集中程度不等于识别准确率，不据此排名。', '',
              '已验证 NPZ/JSON/CSV 概率逐值一致、逐帧概率和为1、站点原始格式校验通过。']
    if browser['verified']:
        lines += ['Firefox 已验证四模型切换、全42类热图、四模型总览可见；当前无 WebGL，未验证3D旋转。']
    else:
        lines += ['当前无显示环境中的 Firefox 自动渲染检查未完成；未宣称浏览器交互已验证。']
    lines += ['', '打开 `posterior_3d_interactive.html`，或把 `buckeye_buckeye_s0701a_0000_seed43.posterior-v1.json` 导入之前的 Posterior Lab。`statistics.json` 含逐类统计。', '',
              '全部数据本地保存，无上传；参考只用于标注，不参与推理。']
    report = '\n'.join(lines) + '\n'
    (output / 'REPORT_ZH.md').write_text(report)
    (HERE / 'REPORT_ZH.md').write_text(report + '\n交付目录：`' + str(output) + '`\n')
    for name in ['README_ZH.md', 'export.py', 'render.py', 'package.py']:
        shutil.copyfile(HERE / name, output / name)
    complete.update(browser_verified=browser['verified'], site_schema_verified=True,
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in output.iterdir() if p.is_file() and p.name != 'COMPLETE.json'})
    (output / 'COMPLETE.json').write_text(json.dumps(complete, indent=2) + '\n')
    archive = HERE / 'buckeye_real_posterior_seed43.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(output.iterdir()):
            if path.is_file():
                stream.write(path, path.name)
    with zipfile.ZipFile(archive) as stream:
        assert stream.testzip() is None
    print(json.dumps(dict(output=str(output), zip=str(archive), zip_bytes=archive.stat().st_size,
                          browser=browser, files=len(complete['files'])), indent=2))


if __name__ == '__main__':
    main()
