"""
Modified from deformable_detr_twostage_refine_r50_16x2_50e_coco.py
"""

_base_ = 'deformable_detr_ghost_freqnet50_refine_16x2_50e_coco.py'
model = dict(bbox_head=dict(as_two_stage=True))
