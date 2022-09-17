import math
import numpy as np
import torch
import torch.nn as nn
import bisect
from dataset.process import *



def deploy_preprocess(output, params):
    ''' snpe 对sigmoid tanh 这些带exp指数的运算优化还没到位 '''
    if params.is_BNC:
        h, w = params.feature_map_size
        c = params.num_feature_map_channel
        output = output.transpose(-2, -1).reshape(-1, c, h, w)
    if params.without_exp:
        # 放到模型外面（后）处理
        # Confidence point_x, point_y
        output[:, :3] = output[:, :3].sigmoid()
        # SepLine(deltanorm or direction) + EntryLine
        output[:, 3:7] = output[:, 3:7].tanh()
        # isOccupied
        output[:, 7] = output[:, 7].sigmoid()
    return output


def compute_losses(j, output, train_batch, params, is_train=False):
    losses = {}
    objective, gradient = generate_objective(j, train_batch, params, is_train)
    predictions = output if is_train else output[j:j + 1]

    # compute losses
    if params.loss_type in ["basic", 'l2']: # mean square error (MSE)
        if params.loss_l2_type == 'multi_v4':
            losses["total"] = calc_l2_multi_v4(predictions, objective)
        elif params.loss_l2_type == 'multi_v2':
            losses["total"] = calc_l2_multi_v2(predictions, objective)
        elif params.loss_l2_type == 'multi_v1':
            losses['total'] = calc_l2_multi_v1(predictions, objective)
        else: # default
            losses["total"] = calc_l2(predictions, objective)
    elif params.loss_type == 'l1': # mean absolute error (MAE)
        losses['total'] = calc_l1(predictions, objective)
    elif params.loss_type == 'smooth-l1':
        losses['total'] = calc_sl1(predictions, objective)
    else:
        raise NotImplementedError

    return losses, gradient


def compute_eval_results(j, data_batch, output, params):
    marking_points = data_batch["MarkingPoint"][j]
    predictions = output[j]

    losses, gradient = compute_losses(j, output, data_batch, params)
    loss = torch.sum(losses["total"] * gradient).item()

    pred_points = get_predicted_points(predictions, params)
    eval_results = collect_error(marking_points, pred_points, params)

    eval_results['loss'] = loss
    eval_results['pred_points'] = pred_points
    return eval_results


def generate_objective_kernel(batch_idx, params, objective,
                              gradient, marking_point):
    # marking_point.x and marking_point.y are in range [0, H-1/H or W-1/W]
    '''
        这里会截断图像外的点，但是是大于像素的点，
        在成像中 就是 右边儿 和 下边儿， 被截断了
        但是，左边和上边的点，小于图像的，被保留了，
        为什么会超出呢，因为，rotate，数据增强时，超出的都保留下来了
        先暂时不改，因为目前的几十个实验，都是基于这个做的，后续再改 (   ) todo fixme
    '''
    H_f, W_f = params.feature_map_size
    H, W = params.feature_map_size
    col_x = math.floor(marking_point.x * W_f)
    row_y = math.floor(marking_point.y * H_f)
    # previous method  exp 0-45
    col = min(col_x, W - 1)
    row = min(row_y, H - 1)  # 小于0的比如-1，成了最大值，error!!!
    # The present method: exp 46 start
    if params.filter_outer_point:
        col = min(max(col_x, 0), W - 1)
        row = min(max(row_y, 0), H - 1)
        # 写 col = col_x, row = row_y 也可，因为dataloader里做了过滤了
        # 还有一种情况是，超过5个像素内的，给予通过，同样是在dataloader里设计
        # ，毕竟一个格子占32个像素，并到边界格子中
        # 。只要小于1个格子的像素，并起来都算合理，但实际上画图出来看，就差别大了
        # ，5个像素几乎没影响，10个像素看得出来，可以做个实验
    # Confidence Regression
    objective[batch_idx, 0, row, col] = 1.  # 在这里！
    # Offset Regression (related position)
    objective[batch_idx, 1, row, col] = marking_point.x * \
                                        params.feature_map_size[1] - col
    objective[batch_idx, 2, row, col] = marking_point.y * \
                                        params.feature_map_size[0] - row
    # method
    if params.deltanorm_or_direction == 'deltanorm':
        # Angle Regression
        objective[batch_idx, 3, row, col] = marking_point.lenSepLine_x
        objective[batch_idx, 4, row, col] = marking_point.lenSepLine_y
    elif params.deltanorm_or_direction == 'direction':
        # Direction Regression
        H, W = params.input_size
        x_ratio, y_ratio = 1, H / W
        radian = math.atan2(marking_point.lenSepLine_y * y_ratio,
                            marking_point.lenSepLine_x * x_ratio)
        objective[batch_idx, 3, row, col] = math.cos(radian)
        objective[batch_idx, 4, row, col] = math.sin(radian)
    # elif: params.deltanorm_or_direction == 'Deltanorm and Direction':
    # 这个要改的东西太多了，就不加到这个project里来了
    else:
        raise NotImplementedError(f'Unkown method: '
                                  f'{params.deltanorm_or_direction}')
    # Entry Line Regression
    objective[batch_idx, 5, row, col] = marking_point.lenEntryLine_x
    objective[batch_idx, 6, row, col] = marking_point.lenEntryLine_y
    objective[batch_idx, 7, row, col] = marking_point.isOccupied
    # Assign Gradient
    gradient[batch_idx, 1:, row, col].fill_(1.)

    return objective, gradient


def generate_objective(j, train_batch, params, is_train=False):
    """Get regression objective and gradient for directional point detector."""
    if is_train:
        marking_points_batch = train_batch['MarkingPoint']
        device = train_batch['image'].device
        batch_size = len(marking_points_batch)
    else: # evaluate
        marking_points_batch = train_batch['MarkingPoint'][j]
        device = train_batch['image'][j].device
        batch_size = 1
        batch_idx = 0
    objective = torch.zeros(batch_size,
                            params.num_feature_map_channel,
                            params.feature_map_size[0],
                            params.feature_map_size[1],
                            device=device)
    gradient = torch.zeros_like(objective)
    gradient[:, 0].fill_(1.) # 不管是有角点还是无角点，标签里都是真实值，为True，置信度为1
    if is_train:
        for batch_idx, marking_points in enumerate(marking_points_batch):
            for marking_point in marking_points:
                objective, gradient = \
                    generate_objective_kernel(
                        batch_idx, params, objective, gradient, marking_point)
    else:
        for marking_point in marking_points_batch:
            if 'batch===1' :
                objective, gradient = \
                    generate_objective_kernel(
                        batch_idx, params, objective, gradient, marking_point)

    return objective, gradient



def calc_l1(pd, gt):
    return abs(pd - gt)

def calc_l2(pd, gt):
    return (pd - gt)**2

def calc_sl1(pd, gt):
    d = abs(pd - gt)
    return 0.5 * d**2 if d < 1 else d - 0.5

def calc_l2_multi_v1(pd, gt):
    occupied = torch.unsqueeze((pd[:, 7] - gt[:, 7])**2 / 100, 1)
    confid = torch.unsqueeze((pd[:, 0] - gt[:, 0])**2 / 10, 1)
    others = (pd[:, 1:7] - gt[:, 1:7]) ** 2
    return torch.cat([confid, others, occupied], dim=1)

def calc_l2_multi_v2(pd, gt):
    occupied = torch.unsqueeze((pd[:, 7] - gt[:, 7])**2 / 100, 1)
    confid = torch.unsqueeze((pd[:, 0] - gt[:, 0])**2 / 10, 1)
    entry = (pd[:, 5:7] - gt[:, 5:7])**2 * 10
    others = (pd[:, 1:5] - gt[:, 1:5])**2
    return torch.cat([confid, others, entry, occupied], dim=1)

def calc_l2_multi_v3(pd, gt):
    ''' used for training deltanorm_and_direction '''
    occupied = torch.unsqueeze((pd[:, 9] - gt[:, 7])**2 / 100, 1)
    confid = torch.unsqueeze((pd[:, 0] - gt[:, 0])**2 / 10, 1)
    entry = (pd[:, 7:9] - gt[:, 7:9])**2 * 10
    others = (pd[:, 1:7] - gt[:, 1:7])**2
    return torch.cat([confid, others, entry, occupied], dim=1)

def calc_l2_multi_v4(pd, gt):
    occupied = torch.unsqueeze((pd[:, 7] - gt[:, 7])**2, 1)
    confid = torch.unsqueeze((pd[:, 0] - gt[:, 0])**2 / 10, 1)
    entry = (pd[:, 5:7] - gt[:, 5:7])**2 * 10
    others = (pd[:, 1:5] - gt[:, 1:5])**2
    return torch.cat([confid, others, entry, occupied], dim=1)

def calc_coord_dist(pd, gt):
    return pow((pd.x - gt.x) ** 2 + (pd.y - gt.y) ** 2, 0.5)

def calc_entry_dist(pd, gt):
    pd_x, pd_y = pd.x + pd.lenEntryLine_x, pd.y + pd.lenEntryLine_y
    gt_x, gt_y = gt.x + gt.lenEntryLine_x, gt.y + gt.lenEntryLine_y
    return pow((pd_x - gt_x) ** 2 + (pd_y - gt_y) ** 2, 0.5)

def tmp_get_radian(pd, gt, mode="sepline"):
    if mode == 'sepline':
        pd_radian = math.atan2(pd.lenSepLine_y,
                               pd.lenSepLine_x)
        gt_radian = math.atan2(gt.lenSepLine_y,
                               gt.lenSepLine_x)
    else: # entryline
        pd_radian = math.atan2(pd.lenEntryLine_y,
                               pd.lenEntryLine_x)
        gt_radian = math.atan2(gt.lenEntryLine_y,
                               gt.lenEntryLine_x)
    return pd_radian, gt_radian

def calc_radian(pd, gt, mode='entryline'):
    if mode == 'entryline':
        pd_r, gt_r = tmp_get_radian(pd, gt, mode="entryline")
    else:
        pd_r, gt_r = tmp_get_radian(pd, gt, mode="sepline")
    diff = abs(pd_r - gt_r)
    return diff if diff < math.pi else 2 * math.pi - diff

def calc_radian_cos_sin(pd, gt, mode='entryline'):
    pd_radian, gt_radian = tmp_get_radian(pd, gt, mode)
    pd_cos, pd_sin = math.cos(pd_radian), math.sin(pd_radian)
    gt_cos, gt_sin = math.cos(gt_radian), math.sin(gt_radian)
    return abs(pd_cos - gt_cos), abs(pd_sin - gt_sin)

def collect_error(ground_truths, predictions, params,
                  calc_loss=calc_l2):
    """Collect errors for those correctly detected slots."""
    confidence_err = []
    available_err = []
    coord_dist_err = []
    point_x_err = []
    point_y_err = []
    entryLine_coord_err = []
    entryLine_radian_err = []
    entryLine_cos_err = []
    entryLine_sin_err = []
    sepLine_radian_err = []
    sepLine_cos_err = []
    sepLine_sin_err = []

    predictions = [
        pred for pred in predictions
        if pred[0] >= params.confid_thresh_for_point # 0.11676871
    ]

    for ground_truth in ground_truths:
        idx = match_gt_with_preds(ground_truth, predictions, params)
        if idx >= 0:
            predicted_confidence = predictions[idx][0]
            predicted = predictions[idx][1]
            # namedtuple::MarkingPoint::格式
            confidence_err.append(abs(predicted_confidence - 1.0))
            available_err.append(abs(predicted.isOccupied - ground_truth.isOccupied))
            coord_dist_err.append(calc_coord_dist(predicted, ground_truth))
            point_x_err.append(abs(predicted.x - ground_truth.x))
            point_y_err.append(abs(predicted.y - ground_truth.y))
            entryLine_coord_err.append(calc_entry_dist(predicted, ground_truth))
            # EntryLine
            entryLine_radian_err.append(calc_radian(predicted, ground_truth, mode='entryline'))
            entry_cos, entry_sin = calc_radian_cos_sin(predicted, ground_truth, 'entryline')
            entryLine_cos_err.append(entry_cos)
            entryLine_sin_err.append(entry_sin)
            # SepLine
            sepLine_radian_err.append(calc_radian(predicted, ground_truth, mode='sepline'))
            sep_cos, sep_sin = calc_radian_cos_sin(predicted, ground_truth, 'sepline')
            sepLine_cos_err.append(sep_cos)
            sepLine_sin_err.append(sep_sin)
            if params.deltanorm_or_direction == 'deltanorm':
                pass # sepLine_coord_err
        else:
            continue
    result = {}
    result['confidence_err'] = confidence_err
    result['available_err'] = available_err
    result['coord_dist_err'] = coord_dist_err
    result['point_x_err'] = point_x_err
    result['point_y_err'] = point_y_err
    result['entryLine_coord_err'] = entryLine_coord_err
    result['entryLine_radian_err'] = entryLine_radian_err
    result['entryLine_cos_err'] = entryLine_cos_err
    result['entryLine_sin_err'] = entryLine_sin_err
    result['sepLine_radian_err'] = sepLine_radian_err
    result['sepLine_cos_err'] = sepLine_cos_err
    result['sepLine_sin_err'] = sepLine_sin_err
    return result


def match_gt_with_preds(ground_truth, predictions, params):
    """Match a ground truth with every predictions and return matched index."""
    max_confidence = 0.
    matched_idx = -1

    for i, pred in enumerate(predictions):
        effect_flag = match_marking_points(ground_truth, pred[1], params)
        if effect_flag and max_confidence < pred[0]:
            max_confidence = pred[0]
            matched_idx = i
    return matched_idx


def get_confidence_list(ground_truths_list, predictions_list, params):
    """Generate a list of confidence of true positives and false positives."""
    assert len(ground_truths_list) == len(predictions_list)
    true_positive_list = []
    false_positive_list = []
    true_ava_pos_list = []
    true_occ_pos_list = []
    num_samples = len(ground_truths_list)
    for i in range(num_samples):
        ground_truths = ground_truths_list[i]
        predictions = predictions_list[i]
        prediction_matched = [False] * len(predictions)
        for ground_truth in ground_truths:
            idx = match_gt_with_preds(ground_truth, predictions, params)
            if idx >= 0:
                prediction_matched[idx] = True
                true_positive_list.append(predictions[idx][0])
                if ground_truth.isOccupied == 1:
                    true_ava_pos_list.append(predictions[idx][0])
                else:
                    true_occ_pos_list.append(predictions[idx][0])
            else:
                true_positive_list.append(.0)
                if ground_truth.isOccupied == 1:
                    true_ava_pos_list.append(0.)
                else:
                    true_occ_pos_list.append(0.)
        for idx, pred_matched in enumerate(prediction_matched):
            if not pred_matched:
                false_positive_list.append(predictions[idx][0])
                
    return true_positive_list, false_positive_list, \
        true_ava_pos_list, true_occ_pos_list


def calc_precision_recall(ground_truths_list, predictions_list, params):
    """Adjust threshold to get mutiple precision recall sample."""

    true_positive_list, false_positive_list, \
    true_ava_pos_list, true_occ_pos_list \
        = get_confidence_list(ground_truths_list, predictions_list, params)

    true_positive_list = sorted(true_positive_list)
    false_positive_list = sorted(false_positive_list)
    true_ava_pos_list = sorted(true_ava_pos_list)
    true_occ_pos_list = sorted(true_occ_pos_list)
    thresholds = sorted(list(set(true_positive_list)))
    recalls = [0.]
    precisions = [0.]

    for thresh in reversed(thresholds):
        if thresh == 0.:
            recalls.append(1.)
            precisions.append(0.)
            break
        false_negatives = bisect.bisect_left(true_positive_list, thresh)
        true_positives = len(true_positive_list) - false_negatives
        true_negatives = bisect.bisect_left(false_positive_list, thresh)
        false_positives = len(false_positive_list) - true_negatives
        recalls.append(true_positives / (true_positives + false_negatives))
        precisions.append(true_positives / (true_positives + false_positives))
    FN = bisect.bisect_left(true_positive_list, params.confid_plot_inference)
    TP = len(true_positive_list) - FN
    TN = bisect.bisect_left(false_positive_list, params.confid_plot_inference)
    FP = len(false_positive_list) - TN
    if sum([TP + FP]) == 0:
        Precision, Recall, Accuracy = 0, 0, 0
    else:
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        Accuracy = (TP + TN) / (TP + FP + TN + FN)
    
    # recall for available parking-slot
    FN_ava = bisect.bisect_left(true_ava_pos_list, params.confid_plot_inference)
    TP_ava = len(true_ava_pos_list) - FN_ava
    FN_occ = bisect.bisect_left(true_occ_pos_list, params.confid_plot_inference)
    TP_occ = len(true_occ_pos_list) - FN_occ
    Recall_ava = 0 if sum([TP_ava, FN_ava]) == 0 else TP_ava / (TP_ava + FN_ava)
    Recall_occ = 0 if sum([TP_occ, FN_occ]) == 0 else TP_occ / (TP_occ + FN_occ)
    
    return precisions, recalls, TP, FP, TN, FN, Precision, Recall, Accuracy, \
        TP_ava, FN_ava, Recall_ava, TP_occ, FN_occ, Recall_occ


def calc_average_precision(precisions, recalls, version="VOC"):

    def VOC_version(precisions, recalls):
        """Calculate average precision defined in VOC contest."""
        total_precision = 0.
        for i in range(11):
            index = next(conf[0] for conf in enumerate(recalls)
                         if conf[1] >= i / 10)
            total_precision += max(precisions[index:])

        return total_precision / 11

    def COCO_version(precisions, recalls):
        """Calculate average precision defined in COCO contest."""
        recalls = np.concatenate(([0.], recalls, [1.]))
        precisions = np.concatenate(([0.], precisions, [0.]))
        # compute the precision envelope
        for i in range(recalls.size - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(recalls[1:] != recalls[:-1])[0]
        averege_precision = np.sum(
            (recalls[i + 1] - recalls[i]) * precisions[i + 1])

        return averege_precision

    assert version in ["VOC", "COCO"]
    if version == "VOC":
        return VOC_version(precisions, recalls)
    elif version == "COCO":
        return COCO_version(precisions, recalls)
