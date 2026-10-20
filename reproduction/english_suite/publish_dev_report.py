"""Expose existing dev PER and segmentation histories without rewriting training runs."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"vendor"))
import wandb_workspaces.reports.v2 as wr

SUITE=Path(__file__).resolve().parent

def main():
    report=wr.Report(entity='kevingenghaopeng-',project='ottc-english-multiphoneset',title='English phonemes: dev PER and segmentation',description='Live LibriSpeech four-head recognition and boundary metrics; final TIMIT summary.')
    runs=wr.Runset(entity='kevingenghaopeng-',project='ottc-english-multiphoneset',name='Libri100 CTC CE upstream JSD',filters='Config("dataset") = "librispeech_100h_multiphoneset"')
    # Use the dataset value recorded in the production run config.
    dataset=json.loads((SUITE.parent/'multihead_libri100/models/ctc/run_config.json').read_text())['dataset']
    runs.filters='Config("dataset") = '+json.dumps(dataset)
    panels=[]
    for head in ['ottc_arpabet','ifmdd_arpabet','mfa_ipa']:
        panels.append(wr.LinePlot(title=head+' dev PER (%)',x='train/epoch',y=['eval/PER_percent/'+head],aggregate=False,smoothing_factor=0,smoothing_type='none'))
    panels.append(wr.LinePlot(title='ZIPA dev token error (%)',x='train/epoch',y=['eval/token_error_percent/zipa_ipa'],aggregate=False,smoothing_factor=0,smoothing_type='none'))
    for head in ['ottc_arpabet','ifmdd_arpabet']:
        panels.append(wr.LinePlot(title=head+' dev boundary precision recall F1 (%)',x='train/epoch',y=['eval/boundary_'+metric+'_20ms/'+head for metric in ['precision','recall','F1']],aggregate=False,smoothing_factor=0,smoothing_type='none'))
    report.blocks=[wr.MarkdownBlock(text='LibriSpeech 原始 dev 指标按 epoch 记录。每条曲线来自正式 CTC / OTTC CE / 上游 jsd run；未补造历史指标。边界参考来自自动对齐。MFA/ZIPA 头没有真实区间参考，不报告伪造的 segmentation 准确率。四头的 phoneset 与覆盖范围不同，应在同一 head 内比较。'),wr.PanelGrid(runsets=[runs],panels=panels)]
    report.save()
    (SUITE/'dev_wandb_report.json').write_text(json.dumps({'url':report.url,'dataset_filter':dataset},indent=2)+'\n')
    print(report.url,flush=True)

if __name__=='__main__':main()
