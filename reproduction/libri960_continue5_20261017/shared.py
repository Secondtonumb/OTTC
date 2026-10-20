"""Reuse the frozen Libri100 training metrics and segmentation callback."""
import bootstrap
from bootstrap import ROOT, HERE
import importlib.util

path=ROOT/'reproduction/libri100_singlehead_20261014/train.py'
spec=importlib.util.spec_from_file_location('parent_libri100_training_helpers',path)
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
parent.HERE=HERE
boundary_matches=parent.boundary_matches
phone_per=parent.phone_per
recognition_metrics=parent.recognition_metrics
preprocess=parent.preprocess
DevelopmentSegmentation=parent.DevelopmentSegmentation
