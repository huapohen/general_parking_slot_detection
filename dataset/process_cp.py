"""Defines related function to process defined data structure."""
import imp
import math
import numpy as np
import torch
import math
import cv2
from collections import namedtuple
from enum import Enum

MarkingPoint = namedtuple('MarkingPoint', ['x',
                                           'y',
                                           'lenSepLine_x',
                                           'lenSepLine_y',
                                           'lenEntryLine_x',
                                           'lenEntryLine_y',
                                           'isOccupied'])

def direction_diff(direction_a, direction_b):
    """Calculate the angle between two direction."""
    diff = abs(direction_a - direction_b)
    return diff if diff < math.pi else 2 * math.pi - diff

def calc_point_direction_angle(point_a, point_b):
    """Calculate angle between direction in rad."""
    return direction_diff(point_a.direction, point_b.direction)

def calc_line_direction_angle(line_gt, line_pd, line_type='entry'):
    if line_type == 'entry':
        direc_a = math.atan2(line_gt.lenEntryLine_y*3, line_gt.lenEntryLine_x)
        direc_b = math.atan2(line_pd.lenEntryLine_y*3, line_pd.lenEntryLine_x)
    elif line_type == 'sep':
        direc_a = math.atan2(line_gt.lenSepLine_y*3, line_gt.lenSepLine_x)
        direc_b = math.atan2(line_pd.lenSepLine_y*3, line_pd.lenSepLine_x)
    elif line_type == 'sep_ent':
        direc_a = math.atan2(line_pd.lenSepLine_y*3, line_pd.lenSepLine_x)
        direc_b = math.atan2(line_pd.lenEntryLine_y*3, line_pd.lenEntryLine_x)
    else:
        raise ValueError(f'Wrong eval type: {line_type}')
    return direction_diff(direc_a, direc_b)

def calc_line_dist(gt, pd, axis_type='x'):
    ''' ground truth & predicted 小于半个车位短边'''
    if axis_type == 'x':
        delta_dist = abs(pd.lenEntryLine_x - gt.lenEntryLine_x)
    elif axis_type == 'y':
        delta_dist = abs(pd.lenEntryLine_y - gt.lenEntryLine_y)
    else:
        raise ValueError(f'Wrong eval type: {axis_type}')
    return delta_dist


def calc_point_squre_dist(point_a, point_b):
    """Calculate distance between two marking points."""
    distx = point_a.x - point_b.x
    disty = point_a.y - point_b.y
    return distx**2 + disty**2


def match_marking_points(ground, predict, params):
    """Determine whether a detected point match ground truth."""
    dist_square = calc_point_squre_dist(ground, predict)
    angle_entryLine = calc_line_direction_angle(ground, predict, 'entry')
    angle_sepLine = calc_line_direction_angle(ground, predict, 'sep')
    angle_sep_entry = calc_line_direction_angle(predict, predict, 'sep_ent')
    dist_entry_x = calc_line_dist(ground, predict, 'x')
    dist_entry_y = calc_line_dist(ground, predict, 'y')
    if 0:
        effect_flag = dist_square < params.squared_distance_thresh \
                  and angle_sepLine < params.direction_angle_thresh \
                  and angle_entryLine < params.direction_angle_thresh
                  # and angle_sep_entry > 15 / 180 * math.pi \
                  # and dist_entry_y < 1 / 6 \
                  # and dist_entry_x < 1 / 4 / 2  # 4个格子 feature map  TODO (  )
                     # 12个格子  # TODO 这些可作为限制条件加到train_loss_function里面去
    if 1:
        effect_flag = dist_square < params.squared_distance_thresh
    if 0:
        effect_flag = dist_square < params.squared_distance_thresh \
            and angle_entryLine < params.direction_angle_thresh \
            and angle_sepLine < params.direction_angle_thresh
    return effect_flag


def non_maximum_suppression(pred_points, params):
    """Perform non-maxmum suppression on marking points."""
    ''' 间隔车位可检测 '''
    suppressed = [False] * len(pred_points)
    for i in range(len(pred_points) - 1):
        for j in range(i + 1, len(pred_points)):
            # 0是置信度，1是marking_point
            i_x = pred_points[i][1].x
            i_y = pred_points[i][1].y
            j_x = pred_points[j][1].x
            j_y = pred_points[j][1].y
            if abs(j_x - i_x) < 1 / params.feature_map_size[1] and abs(
                    j_y - i_y) < 1 / params.feature_map_size[0]:
                idx = i if pred_points[i][0] < pred_points[j][0] else j
                suppressed[idx] = True
    if any(suppressed):
        unsupres_pred_points = []
        for i, supres in enumerate(suppressed):
            if not supres:
                unsupres_pred_points.append(pred_points[i])
        return unsupres_pred_points
    return pred_points


def get_predicted_points(prediction, params):
    """Get marking points from one predicted feature map."""
    # 传进来的batch=1
    assert isinstance(prediction, torch.Tensor)
    predicted_points = []
    prediction = prediction.detach().cpu().numpy()
    C, feature_H, feature_W = prediction.shape[-3:]
    assert C == 8
    thresh = params.confid_thresh # 0.01
    for i in range(feature_H):
        for j in range(feature_W):
            if prediction[0, i, j] >= thresh:
                # 这个格子的置信度大于阈值，取其中回归的信息
                # feature格子的归一化
                obj_x = (j + prediction[1, i, j]) / feature_W
                obj_y = (i + prediction[2, i, j]) / feature_H
                lenSepLine_x = prediction[3, i, j]
                lenSepLine_y = prediction[4, i, j]
                lenEntryLine_x = prediction[5, i, j]
                lenEntryLine_y = prediction[6, i, j]
                isOccupied = prediction[7, i, j]
                marking_point = MarkingPoint(obj_x, obj_y,
                                             lenSepLine_x, lenSepLine_y,
                                             lenEntryLine_x, lenEntryLine_y,
                                             isOccupied)
                predicted_points.append((prediction[0, i, j], marking_point))
    if params.is_BNC: # 不需要NMS
        return predicted_points
    else:
        return non_maximum_suppression(predicted_points, params)

# 对于小范围内的多个值，只取置信度最高的那一个(为啥不做加权?)


def plot_slots(image, eval_results, params, img_name=None):
    """
    画进入线目标点：逆时针旋转车位的进入线起始端点。
    AB-BC-CD-DA 这里指A点
       Parking Slot Example
            A.____________D
             |           |
           ==>           |
            B____________|C

     Entry_line: AB
     Separable_line: AD (regressive)
     Separable_line: BC (un-regressive, calc)
     Object_point: A (point_0)
      cos sin theta 依据的笛卡尔坐标四象限
               -y
                |
             3  |  4
        -x -----|-----> +x (w)
             2  |  1
                ↓
               +y (h)
    """
    '''
    cv2 的画图，会把图像外的坐标点，滑倒图像边界，自动抹去，
    所以，即便是预测到了图像外的角点，画的时候，也只能画到图像边界，
    显示会有问题。 需要进一步修复 (   ) TODO
    rotate的warp后 会有黑色区域出现, AVM视频成像：解。
    '''
    pred_points =  eval_results['pred_points'] \
        if 'pred_points' in eval_results else eval_results
    if not pred_points:
        return image
    height = image.shape[0]
    width = image.shape[1]
    for confidence, marking_point in pred_points:
        if confidence < params.confid_plot_inference:
            continue
        x, y, lenSepLine_x, lenSepLine_y, \
        lenEntryLine_x, lenEntryLine_y, isOccupied = marking_point
        # p0->p1为进入线entry_line
        # p0->p3为分隔线separable_line
        # p1->p3也为分割线
        # 上述箭头"->"代表向量方向，p0->p1即p0为起点，p3为终点，p0指向p3
        p0_x = width * x
        p0_y = height * y
        p1_x = p0_x + width * lenEntryLine_x
        p1_y = p0_y + height * lenEntryLine_y
        if params.deltanorm_or_direction == 'deltanorm':
            p3_x = round(p0_x + width * lenSepLine_x)
            p3_y = round(p0_y + height * lenSepLine_y)
            p2_x = round(p1_x + width * lenSepLine_x)
            p2_y = round(p1_y + height * lenSepLine_y)
        elif params.deltanorm_or_direction == 'direction':
            H, W = params.input_size
            x_ratio, y_ratio = 1, H / W
            radian = math.atan2(marking_point.lenSepLine_y * y_ratio,
                                marking_point.lenSepLine_x * x_ratio)
            sep_cos = math.cos(radian)
            sep_sin = math.sin(radian)
            p3_x = round(p0_x + 80 * sep_cos)
            p3_y = round(p0_y + 80 * sep_sin)
            p2_x = round(p1_x + 80 * sep_cos)
            p2_y = round(p1_y + 80 * sep_sin)
        else:
            raise NotImplementedError(f'Unkown method: '
                                      f'{params.deltanorm_or_direction}')
        p0_x, p0_y = round(p0_x), round(p0_y)
        p1_x, p1_y = round(p1_x), round(p1_y)
        # 画进入线目标点：逆时针旋转车位的进入线起始端点。
        # AB-BC-CD-DA 这里指A点
        cv2.circle(image, (p0_x, p0_y), 5, (0, 0, 255), thickness=2)
        # 给目标点打上置信度，取值范围：0到1
        color = (255, 255, 255) if confidence > 0.7 else (100, 100, 255)
        if confidence < 0.3: color = (0, 0, 255)
        cv2.putText(image, f'{confidence:.3f}', # 不要四舍五入
                    (p0_x + 6, p0_y - 4),
                    cv2.FONT_HERSHEY_PLAIN, 1, color)
        # 画上目标点坐标 (x, y)
        cv2.putText(image, f' ({p0_x},{p0_y})',
                    (p0_x, p0_y + 15),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        # 在图像左上角给出图像的分辨率大小 (W, H)
        H, W = params.input_size
        # cv2.putText(image, f'({W},{H})', (5, 15),
        #         cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        # 画车位掩码图，分三步
        # 第一步：画未被占概率，画是否可用提示
        avail = 'N' if isOccupied > 0.9 else 'Y'
        cv2.putText(image, avail,
                    ((p0_x + p1_x) // 2 + 10, (p0_y + p1_y) // 2 + 10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        area = np.array([[p0_x, p0_y], [p1_x, p1_y],
                        [p2_x, p2_y], [p3_x, p3_y]])
        if avail == 'Y':
            color = (0, 255, 0)  # 置信度颜色
            if  1 - isOccupied < 0.5:
                color = (0, 0, 255) # 低置信度红色提示
            cv2.putText(image,
                        f'{1 - isOccupied:.3f}',  # 转换为"未被占用"置信度
                        ((p0_x + p2_x) // 2, (p0_y + p2_y) // 2 + 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, color)
        # 第二步：判断，如果可用，才画掩码图，否则全零
        mask = np.zeros_like(image)
        if 1 - isOccupied > 0.9:
            cv2.fillPoly(mask, [area], [0, 64, 0])
        # 第三步：把掩码图和原图进行叠加： output= a*i1+b*i2+c
        image = cv2.addWeighted(image, 1., mask, 0.5, 0) # 数字是权重
        # 画车位进入线 AB
        cv2.arrowedLine(image, (p0_x, p0_y), (p1_x, p1_y), (0, 255, 0), 2, 8, 0, 0.2)
        # 画车位分割线 AD
        cv2.arrowedLine(image, (p0_x, p0_y), (p3_x, p3_y), (255, 0, 0), 2) # cv2.line
        # 画车位分割线 BC
        if p1_x >= 0 and p1_x <= width - 1 and p1_y >= 0 and p1_y <= height - 1:
            cv2.arrowedLine(image, (p1_x, p1_y), (p2_x, p2_y), (33, 164, 255), 2)

    return image



def rectify_slot_point(pred_points, params, img_name):
    pred_points = rectify_collinear_slot_point(pred_points, params, img_name)
    pred_points = rectify_terrace_slot_point(pred_points, params, img_name)
    pred_points = rectify_annulus_slot_point(pred_points, params, img_name)
    return pred_points

def rectify_collinear_slot_point(pred_points, params, img_name):
    """ 修正共线越位的点 """
    confids = list(list(zip(*pred_points))[0])
    marking_points = list(list(zip(*pred_points))[1])
    new_confids = []
    new_marking_points = []
    H, W = params.input_size
    num_detected = len(marking_points)
    for i in range(num_detected):
        if params.filter_outer_point:
            px = marking_points[i].x * W
            py = marking_points[i].y * H
            gap = 10 # 超出10个像素以外的点，就不要画了
            if px < -gap or py < -gap or px > W + gap or py > H + gap:
                continue
        new_confids.append(confids[i])
        rectify_sign = 0
        for j in range(i + 1, num_detected):
            p1 = marking_points[i]
            p2 = marking_points[j]
            vec1 = np.array([W * p1.lenEntryLine_x, H * p1.lenEntryLine_y])
            vec2 = np.array([W * p2.lenEntryLine_x, H * p2.lenEntryLine_y])
            vec12 = np.array([W * (p2.x - p1.x), H * (p2.y - p1.y)])
            len_vec1 = np.linalg.norm(vec1)
            len_vec2 = np.linalg.norm(vec2)
            len_vec12 = np.linalg.norm(vec12)
            vec1 = vec1 / len_vec1
            vec2 = vec2 / len_vec2
            vec12 = vec12 / len_vec12
            collinear_or_not = np.dot(vec1, vec12) > 0.8 # super-parameter
            if len_vec1 > len_vec12 \
                    and collinear_or_not:
                    # and len_vec1 < len_vec12 + len_vec2 * 4 / 5 \
                p1 = p1._replace(lenEntryLine_x = p2.x - p1.x)
                p1 = p1._replace(lenEntryLine_y = p2.y - p1.y)
                new_marking_points.append(p1)
                rectify_sign = 1
        if rectify_sign == 0:
            new_marking_points.append(marking_points[i])

    return list(zip(new_confids, new_marking_points))

def rectify_terrace_slot_point(pred_points, params, img_name):
    """ 修正梯田越位的点 """
    # 两线交点  无需修正，效果很好，网络直出
    return pred_points

def rectify_annulus_slot_point(pred_points, params, img_name):
    """ 修正环形越位的点 """
    return pred_points