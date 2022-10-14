_base_ = 'deformable_detr_ghost_freqnet50_metaacon_16x2_50e_coco.py'
model = dict(bbox_head=dict(with_box_refine=True))
