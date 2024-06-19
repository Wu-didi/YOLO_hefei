import torch.nn as nn
import torch
from .general import bbox_iou
from .postprocess import build_targets, build_targets_multi
from lib.core.evaluate import SegmentationMetric
import torch.nn.functional as F


def l1_loss_depth(input, target, val=False):
    mask = target > 0
    if mask.data.sum() < 1:
        return None

    lss = F.l1_loss(input[mask], target[mask], reduction="mean")
    return lss

class MultiHeadLoss_origin(nn.Module):
    """
    collect all the loss we need
    """
    def __init__(self, losses, cfg, lambdas=None):
        """
        Inputs:
        - losses: (list)[nn.Module, nn.Module, ...]
        - cfg: config object
        - lambdas: (list) + IoU loss, weight for each loss
        """
        super().__init__()
        # lambdas: [cls, obj, iou, la_seg, ll_seg, ll_iou]
        if not lambdas:
            lambdas = [1.0 for _ in range(len(losses) + 3)]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = nn.ModuleList(losses)
        self.lambdas = lambdas
        self.cfg = cfg

    def forward(self, head_fields, head_targets, shapes, model):
        """
        Inputs:
        - head_fields: (list) output from each task head
        - head_targets: (list) ground-truth for each task head
        - model:

        Returns:
        - total_loss: sum of all the loss
        - head_losses: (tuple) contain all loss[loss1, loss2, ...]

        """
        # head_losses = [ll
        #                 for l, f, t in zip(self.losses, head_fields, head_targets)
        #                 for ll in l(f, t)]
        #
        # assert len(self.lambdas) == len(head_losses)
        # loss_values = [lam * l
        #                for lam, l in zip(self.lambdas, head_losses)
        #                if l is not None]
        # total_loss = sum(loss_values) if loss_values else None
        # print(model.nc)
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)

        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes, model):
        """

        Args:
            predictions: predicts of [[det_head1, det_head2], drive_area_seg_head, lane_line_seg_head]
            targets: gts [det_targets, segment_targets, lane_targets]
            model:

        Returns:
            total_loss: sum of all the loss
            head_losses: list containing losses

        """
        cfg = self.cfg
        device = targets[0].device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = build_targets(cfg, predictions[0], targets[0], model)  # targets

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)

        BCEcls, BCEobj,_ = self.losses

        # Calculate Losses
        nt = 0  # number of targets
        no = len(predictions[0])  # number of outputs
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6

        # calculate detection loss
        for i, pi in enumerate(predictions[0]):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                # print(model.nc)
                if model.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE
            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss



        nb, _, height, width = targets[1].shape
        pad_w, pad_h = shapes[0][1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        
        depth_pred = predictions[1][:, :, pad_h:height-pad_h, pad_w:width-pad_w]
        depth_gt = targets[1][:, :, pad_h:height-pad_h, pad_w:width-pad_w]
        depth_loss = l1_loss_depth(depth_pred, depth_gt)
        
        # 计算深度估计mse指标
        mse_loss_fn = nn.MSELoss()
        assert depth_pred.shape == depth_gt.shape, "y_true and y_pred should have the same shape" 
        mse_loss = mse_loss_fn(depth_pred, depth_gt)  
        
        
        

        loss = lbox + lobj + lcls + depth_loss + mse_loss 
        # loss = lseg
        # return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
        return loss, (lbox.item(), lobj.item(), lcls.item(), depth_loss.item(), mse_loss, loss.item())


class MultiHeadLoss(nn.Module):
    """
    collect all the loss we need
    """
    def __init__(self, losses, cfg, lambdas=None):
        """
        Inputs:
        - losses: (list)[nn.Module, nn.Module, ...]
        - cfg: config object
        - lambdas: (list) + IoU loss, weight for each loss
        """
        super().__init__()
        # lambdas: [cls, obj, iou, la_seg, ll_seg, ll_iou]
        if not lambdas:
            lambdas = [1.0 for _ in range(len(losses) + 3)]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = nn.ModuleList(losses)
        self.lambdas = lambdas
        self.cfg = cfg

    def forward(self, head_fields, head_targets, shapes, model):
        """
        Inputs:
        - head_fields: (list) output from each task head
        - head_targets: (list) ground-truth for each task head
        - model:

        Returns:
        - total_loss: sum of all the loss
        - head_losses: (tuple) contain all loss[loss1, loss2, ...]

        """
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)

        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes, model):
        """

        Args:
            predictions: predicts of [[det_head1, det_head2], drive_area_seg_head, lane_line_seg_head]
            targets: gts [det_targets, segment_targets, lane_targets]
            model:

        Returns:
            total_loss: sum of all the loss
            head_losses: list containing losses

        """
        det_loss_0, head_losses_0 = self._forward_one_det(predictions[0], targets[0], shapes, model, 0)
        det_loss_1, head_losses_1 = self._forward_one_det(predictions[1], targets[1], shapes, model, 1)
        det_loss_2, head_losses_2 = self._forward_one_det(predictions[2], targets[2], shapes, model, 2)
        
        # det_loss = det_loss_0 + det_loss_1 + det_loss_2
        det_loss = det_loss_0
        return det_loss, [head_losses_0, head_losses_1, head_losses_2]


    def _forward_one_det(self, predictions, targets, shapes, model, head_index):
        """

        Args:
            predictions: predicts of [[det_head1, det_head2], drive_area_seg_head, lane_line_seg_head]
            targets: gts [det_targets, segment_targets, lane_targets]
            model:

        Returns:
            total_loss: sum of all the loss
            head_losses: list containing losses

        """
        cfg = self.cfg
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = build_targets_multi(cfg, predictions, targets, model, head_index)  # targets

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)

        BCEcls, BCEobj,_ = self.losses

        # Calculate Losses
        nt = 0  # number of targets
        no = len(predictions)  # number of outputs
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6

        # calculate detection loss
        for i, pi in enumerate(predictions):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                # print(model.nc)
                if model.nc > 1:  # cls loss (only if multiple classes)
                # if True:
                    # print("tcls", tcls[i])
                    t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE
            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss
        loss = lbox + lobj + lcls 
        
        # loss = lseg
        # return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
        return loss, (lbox.item(), lobj.item(), lcls.item())



def get_loss(cfg, device):
    """
    get MultiHeadLoss

    Inputs:
    -cfg: configuration use the loss_name part or 
          function part(like regression classification)
    -device: cpu or gpu device

    Returns:
    -loss: (MultiHeadLoss)

    """
    # class loss criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.CLS_POS_WEIGHT])).to(device)
    # object loss criteria
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.OBJ_POS_WEIGHT])).to(device)
    # segmentation loss criteria
    BCEseg = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)
    # Focal loss
    gamma = cfg.LOSS.FL_GAMMA  # focal loss gamma
    if gamma > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma)

    loss_list = [BCEcls, BCEobj, BCEseg]
    loss = MultiHeadLoss(loss_list, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)
    return loss

# example
# class L1_Loss(nn.Module)


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        # alpha  balance positive & negative samples
        # gamma  focus on difficult samples
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
