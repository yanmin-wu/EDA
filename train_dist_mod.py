# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------
"""Main script for language modulation."""

import os

import numpy as np
import torch
import torch.distributed as dist

from main_utils import parse_option, BaseTrainTester
from data.model_util_scannet import ScannetDatasetConfig
from src.joint_det_dataset import Joint3DDataset
from src.grounding_evaluator import GroundingEvaluator
from models import BeaUTyDETR
from models import APCalculator, parse_predictions, parse_groundtruths

from tqdm import tqdm
import datetime

import ipdb
st = ipdb.set_trace

import numpy as np

class TrainTester(BaseTrainTester):
    """Train/test a language grounder."""

    # logger.
    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

    # BRIEF Initialize dataset.
    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""

        dataset_dict = {}  # dict to use multiple datasets
        for dset in args.dataset:
            dataset_dict[dset] = 1
        if args.joint_det:
            dataset_dict['scannet'] = 10
        print('Loading datasets:', sorted(list(dataset_dict.keys())))

        if args.eval:
            train_dataset = None
        else:
            train_dataset = Joint3DDataset(
                dataset_dict=dataset_dict,
                test_dataset=args.test_dataset,
                split='train' if not args.debug else 'val',
                use_color=args.use_color, use_height=args.use_height,
                overfit=args.debug,
                data_path=args.data_root,
                detect_intermediate=args.detect_intermediate,
                use_multiview=args.use_multiview,
                butd=args.butd,
                butd_gt=args.butd_gt,
                butd_cls=args.butd_cls,
                augment_det=args.augment_det
            )
        
        test_dataset = Joint3DDataset(
            dataset_dict=dataset_dict,
            test_dataset=args.test_dataset,
            split='val' if not args.eval_train else 'train',
            use_color=args.use_color, use_height=args.use_height,
            overfit=args.debug,
            data_path=args.data_root,
            detect_intermediate=args.detect_intermediate,
            use_multiview=args.use_multiview,
            butd=args.butd,
            butd_gt=args.butd_gt,
            butd_cls=args.butd_cls,
            wo_obj_name=args.wo_obj_name
        )
        return train_dataset, test_dataset

    # BRIEF Initialize the model.
    @staticmethod
    def get_model(args):
        """Initialize the model."""
        num_input_channel = int(args.use_color) * 3
        if args.use_height:
            num_input_channel += 1
        if args.use_multiview:
            num_input_channel += 128
        if args.use_soft_token_loss:
            num_class = 256
        else:
            num_class = 19
        model = BeaUTyDETR(
            num_class=num_class,
            num_obj_class=485,
            input_feature_dim=num_input_channel,
            num_queries=args.num_target,
            num_decoder_layers=args.num_decoder_layers,
            self_position_embedding=args.self_position_embedding,
            contrastive_align_loss=args.use_contrastive_align,
            butd=args.butd or args.butd_gt or args.butd_cls,
            pointnet_ckpt=args.pp_checkpoint,
            data_path = args.data_root,
            self_attend=args.self_attend
        )
        return model

    # BRIEF input data.
    @staticmethod
    def _get_inputs(batch_data):
        return {
            'point_clouds': batch_data['point_clouds'].float(), # ([B, 50000, 6]) xyz + colour
            'text': batch_data['utterances'],                   # list[B]  text
            "det_boxes": batch_data['all_detected_boxes'],      # ([B, 132, 6]) groupfree detection boxes
            "det_bbox_label_mask": batch_data['all_detected_bbox_label_mask'],  # ([B, 132]) mask
            "det_class_ids": batch_data['all_detected_class_ids']   # ([B, 132])  box id
        }


    # BRIEF only eval one epoch.
    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader,
                           model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        # [Option] Object detection evaluation on ScanNet dataset.
        if args.test_dataset == 'scannet':      
            return self.evaluate_one_epoch_det(
                epoch, test_loader, model,
                criterion, set_criterion, args
            )

        stat_dict = {}
        model.eval()  # set model to eval mode (for bn and dp)
        # 7 layers: proposal, last, 0-4
        if args.num_decoder_layers > 0:
            prefixes = ['last_', 'proposal_']
            prefixes = ['last_']
            prefixes.append('proposal_')
        else:
            prefixes = ['proposal_']  # only proposal
        prefixes += [f'{i}head_' for i in range(args.num_decoder_layers - 1)]

        evaluator = GroundingEvaluator(
            only_root=True, thresholds=[0.25, 0.5],     # TODO only_root=True
            topks=[1, 5, 10], prefixes=prefixes,
            filter_non_gt_boxes=args.butd_cls
        )

        # NOTE Main eval branch
        test_loader = tqdm(test_loader)
        for batch_idx, batch_data in enumerate(test_loader):
            # note forward and compute loss
            stat_dict, end_points = self._main_eval_branch(     
                batch_idx, batch_data, test_loader, model, stat_dict,
                criterion, set_criterion, args
            )
            if evaluator is not None:
                for prefix in prefixes:
                    # note only consider the last layer
                    if prefix != 'last_':
                        continue

                    # evaluation
                    evaluator.evaluate(end_points, prefix)      

        evaluator.synchronize_between_processes()
        if dist.get_rank() == 0:
            if evaluator is not None:
                # tensorboard eval socre
                s_25 = evaluator.dets[("last_", 0.25, 1, "bbs")] / max(evaluator.gts[("last_", 0.25, 1, "bbs")], 1)
                s_50 = evaluator.dets[("last_", 0.5,  1, "bbs")] / max(evaluator.gts[("last_", 0.5,  1, "bbs")], 1)
                c_25 = evaluator.dets[("last_", 0.25, 1, "bbf")] / max(evaluator.gts[("last_", 0.25, 1, "bbf")], 1)
                c_50 = evaluator.dets[("last_", 0.5,  1, "bbf")] / max(evaluator.gts[("last_", 0.5,  1, "bbf")], 1)
                self.tensorboard.item["val_score"]["soft_token_0.25"] = s_25
                self.tensorboard.item["val_score"]["soft_token_0.5"] = s_50
                self.tensorboard.item["val_score"]["contrastive_0.25"] = c_25
                self.tensorboard.item["val_score"]["contrastive_0.5"] = c_50
                self.tensorboard.dump_tensorboard("val_score", epoch)
                # tensorboard eval loss
                for key in self.tensorboard.item["val_loss"]:
                    self.tensorboard.item["val_loss"][key] = stat_dict[key] / len(test_loader)
                self.tensorboard.dump_tensorboard("val_loss", epoch)

                evaluator.print_stats()
        return None
    
    # BRIEF Scannet detection evalution
    @torch.no_grad()
    def evaluate_one_epoch_det(self, epoch, test_loader,
                               model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        dataset_config = ScannetDatasetConfig(18)
        # Used for AP calculation
        CONFIG_DICT = {
            'remove_empty_box': False, 'use_3d_nms': True,
            'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
            'per_class_proposal': True, 'conf_thresh': 0.0,
            'dataset_config': dataset_config,
            'hungarian_loss': True
        }
        stat_dict = {}
        model.eval()  # set model to eval mode (for bn and dp)
        if set_criterion is not None:
            set_criterion.eval()

        if args.num_decoder_layers > 0:
            prefixes = ['last_', 'proposal_']
            prefixes += [
                f'{i}head_' for i in range(args.num_decoder_layers - 1)
            ]
        else:
            prefixes = ['proposal_']  # only proposal
        prefixes = ['last_']
        ap_calculator_list = [
            APCalculator(iou_thresh, dataset_config.class2type)
            for iou_thresh in args.ap_iou_thresholds
        ]
        mAPs = [
            [iou_thresh, {k: 0 for k in prefixes}]
            for iou_thresh in args.ap_iou_thresholds
        ]

        batch_pred_map_cls_dict = {k: [] for k in prefixes}
        batch_gt_map_cls_dict = {k: [] for k in prefixes}

        # Main eval branch
        # NOTE char span and token span.
        wordidx = np.array([
            0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11,
            12, 13, 13, 14, 15, 16, 16, 17, 17, 18, 18
        ])  # 18+1（not mentioned）
        tokenidx = np.array([
            1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19, 21, 23,
            25, 27, 29, 31, 32, 34, 36, 38, 39, 41, 42, 44, 45
        ])  # 18 token span

        test_loader = tqdm(test_loader)
        for batch_idx, batch_data in enumerate(test_loader):
            # note eval
            stat_dict, end_points = self._main_eval_branch(
                batch_idx, batch_data, test_loader, model, stat_dict,
                criterion, set_criterion, args
            )

            # step score   contrast
            proj_tokens = end_points['proj_tokens']  # (B, tokens, 64)
            proj_queries = end_points['last_proj_queries']  # (B, Q, 64)
            sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
            sem_scores_ = sem_scores / 0.07  # (B, Q, tokens)
            sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256)
            sem_scores = sem_scores.to(sem_scores_.device)
            sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_
            end_points['last_sem_cls_scores'] = sem_scores  # ([B, 256, 256])

            # step
            sem_cls = torch.zeros_like(end_points['last_sem_cls_scores'])[..., :19] # ([B, 256, 19])
            for w, t in zip(wordidx, tokenidx):
                sem_cls[..., w] += end_points['last_sem_cls_scores'][..., t]
            end_points['last_sem_cls_scores'] = sem_cls     # ([B, 256, 19])

            # step Parse predictions
            # for prefix in prefixes:
            prefix = 'last_'
            # pred
            batch_pred_map_cls = parse_predictions(
                end_points, CONFIG_DICT, prefix,
                size_cls_agnostic=True)
            batch_gt_map_cls = parse_groundtruths(
                end_points, CONFIG_DICT,
                size_cls_agnostic=True)
            batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
            batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

        mAP = 0.0
        # for prefix in prefixes:
        prefix = 'last_'
        for (batch_pred_map_cls, batch_gt_map_cls) in zip(
                batch_pred_map_cls_dict[prefix],
                batch_gt_map_cls_dict[prefix]):
            for ap_calculator in ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
        
        # Evaluate average precision
        for i, ap_calculator in enumerate(ap_calculator_list):
            metrics_dict = ap_calculator.compute_metrics()
            self.logger.info(
                '=====================>'
                f'{prefix} IOU THRESH: {args.ap_iou_thresholds[i]}'
                '<====================='
            )
            for key in metrics_dict:
                self.logger.info(f'{key} {metrics_dict[key]}')
            if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
                mAP = metrics_dict['mAP']
            mAPs[i][1][prefix] = metrics_dict['mAP']
            ap_calculator.reset()

        for mAP in mAPs:
            self.logger.info(
                f'IoU[{mAP[0]}]:\t'
                + ''.join([
                    f'{key}: {mAP[1][key]:.4f} \t'
                    for key in sorted(mAP[1].keys())
                ])
            )

        return None


if __name__ == '__main__':
    # huggingface
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    opt = parse_option()    
    
    # distributed 
    torch.cuda.set_device(opt.local_rank)
    # https://github.com/open-mmlab/mmcv/issues/1969#issuecomment-1304721237
    torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400))  
    
    # cudnn
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    train_tester = TrainTester(opt)
    ckpt_path = train_tester.main(opt)
