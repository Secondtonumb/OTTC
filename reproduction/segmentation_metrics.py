"""R-value reporting from recorded P/R using the frozen IF-MDD implementation.

This leaves each evaluator's boundary extraction/matching protocol unchanged.
Adding R-value does not make different boundary protocols interchangeable.
"""
import importlib.util
from pathlib import Path

SOURCE=Path(__file__).resolve().parent/'english_suite/mfa_protocol_comparison/scorer_source/boundary_rvalue.py'
spec=importlib.util.spec_from_file_location('frozen_ifmdd_r_value',SOURCE)
_scorer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(_scorer)


def r_value_percent(precision_percent,recall_percent):
    if precision_percent is None or recall_percent is None:return None
    return 100*_scorer.r_value(float(precision_percent)/100,float(recall_percent)/100)[0]


def boundary_r_value(boundary):
    if not boundary:return None
    if boundary.get('reference')==0:return None
    if 'r_value_percent' in boundary:return boundary['r_value_percent']
    if 'r_value' in boundary:return 100*boundary['r_value']
    precision=boundary.get('precision_percent')
    recall=boundary.get('recall_percent')
    if precision is None and 'precision' in boundary:precision=100*boundary['precision']
    if recall is None and 'recall' in boundary:recall=100*boundary['recall']
    return r_value_percent(precision,recall)


def flat_boundary_r_value(metrics,prefix='',head=None,tolerance=20):
    if metrics.get('reference_boundaries')==0:return None
    stem=f'{prefix}boundary_'
    suffix=f'_{tolerance}ms'+(f'/{head}' if head is not None else '_percent')
    key=stem+'R_value'+suffix
    if key in metrics:return metrics[key]
    return r_value_percent(metrics.get(stem+'precision'+suffix),metrics.get(stem+'recall'+suffix))


def enrich_flat_metrics(metrics):
    """Add derived R-value fields wherever flat recorded P/R fields exist."""
    result=dict(metrics)
    if metrics.get('reference_boundaries')==0:return result
    for key,precision in metrics.items():
        if 'boundary_precision_' not in key:continue
        recall_key=key.replace('boundary_precision_','boundary_recall_',1)
        if recall_key not in metrics:continue
        value=r_value_percent(precision,metrics[recall_key])
        if value is not None:
            result[key.replace('boundary_precision_','boundary_R_value_',1)]=value
    return result
