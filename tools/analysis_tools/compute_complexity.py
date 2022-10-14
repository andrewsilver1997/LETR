
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from mmcv.cnn import get_model_complexity_info
from mmcv import Config
from mmdet.models import build_detector


def main():
    input_shape = (3, 1333, 800)
    cfg_path = 'configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py'
    cfg = Config.fromfile(cfg_path)
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()