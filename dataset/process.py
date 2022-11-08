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


def calc_point_squre_dist_v1(point_a, point_b):
    """Calculate distance between two marking points."""
    distx = point_a.x - point_b.x
    disty = point_a.y - point_b.y
    # res = (distx/3)**2 + (disty*1)**2
    # res = distx**2 + (disty*1)**2
    res = distx**2 + (disty*3)**2 # H:W = 3:1
    return res


def calc_point_squre_dist_v2(gt, pd):
    gt_x = gt.x + gt.lenEntryLine_x
    gt_y = gt.y + gt.lenEntryLine_y
    pd_x = pd.x + pd.lenEntryLine_x
    pd_y = pd.y + pd.lenEntryLine_y
    # y * 3, 相对 x 来说，以短边为基准；更严格
    return (gt_x - pd_x)**2 + (gt_y*3 - pd_y*3)**2


def match_marking_points(ground, predict, params):
    """Determine whether a detected point match ground truth."""
    entry_startpoint = calc_point_squre_dist_v1(ground, predict)
    entry_endpoint   = calc_point_squre_dist_v2(ground, predict)
    angle_entry = calc_line_direction_angle(ground, predict, 'entry')
    angle_sep   = calc_line_direction_angle(ground, predict, 'sep')
    # 3 for 30 degrees and 1 for 10 degrees
    ratio = 1
    # ratio = 2
    # ratio = 3
    # ratio = 6
    # mode 1 strict
    effect_flag = entry_startpoint < params.squared_distance_thresh \
                  and entry_endpoint < params.squared_distance_thresh \
                  and angle_sep < params.direction_angle_thresh / ratio \
                  and angle_entry < params.direction_angle_thresh / ratio
    # mode 2 loose
    effect_flag = entry_startpoint < params.squared_distance_thresh
    # effect_flag = entry_endpoint < params.squared_distance_thresh
    # effect_flag = entry_startpoint < params.squared_distance_thresh \
    #               and angle_sep < params.direction_angle_thresh / ratio
    # effect_flag = entry_startpoint < params.squared_distance_thresh \
    #               and entry_endpoint < params.squared_distance_thresh
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

    return non_maximum_suppression(predicted_points, params)
    # if params.is_BNC: # 不需要NMS
    #     return predicted_points
    # else:
    #     return non_maximum_suppression(predicted_points, params)


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
    pred_points =  eval_results['pred_points'] \
        if 'pred_points' in eval_results else eval_results
    if not pred_points:
        return image
    height = image.shape[0]
    width = image.shape[1]
    for confidence, marking_point in pred_points:
        if confidence < params.confid_plot_inference:
            continue # DMPR-PS是0.5
        x, y, lenSepLine_x, lenSepLine_y, \
        lenEntryLine_x, lenEntryLine_y, available = marking_point
        # p0->p1为进入线entry_line
        # p0->p3为分隔线separable_line
        # p1->p3也为分割线
        # 上述箭头"->"代表向量方向，p0->p1即p0为起点，p3为终点，p0指向p3
        p0_x = width * x - 1
        p0_y = height * y - 1
        p1_x = p0_x + width * lenEntryLine_x
        p1_y = p0_y + height * lenEntryLine_y
        length = 300
        if params.deltanorm_or_direction == 'deltanorm':
            p3_x = int(p0_x + width * lenSepLine_x)
            p3_y = int(p0_y + height * lenSepLine_y)
            p2_x = int(p1_x + width * lenSepLine_x)
            p2_y = int(p1_y + height * lenSepLine_y)
        elif params.deltanorm_or_direction == 'direction':
            H, W = params.input_size
            x_ratio, y_ratio = 1, H / W
            radian = math.atan2(marking_point.lenSepLine_y * y_ratio,
                                marking_point.lenSepLine_x * x_ratio)
            sep_cos = math.cos(radian)
            sep_sin = math.sin(radian)
            p3_x = int(p0_x + length * sep_cos)
            p3_y = int(p0_y + length * sep_sin)
            p2_x = int(p1_x + length * sep_cos)
            p2_y = int(p1_y + length * sep_sin)
            if params.direction_rectangle:  # fixed rectangle slot
                vec_entry = np.array([lenEntryLine_x * width,
                                      lenEntryLine_y * height])
                vec_entry = vec_entry / np.linalg.norm(vec_entry)
                vec_sep = rotate_vector(vec_entry, 90)
                if params.pd_or_gt == 'ground truth':
                    vec_sep = [sep_cos, sep_sin] # not rectangle
                    p3_x = p0_x + length * vec_sep[0]
                    p3_y = p0_y + length * vec_sep[1]
                    p2_x = p1_x + length * vec_sep[0]
                    p2_y = p1_y + length * vec_sep[1]
                else:
                    len_ratio = 2.27 # long/short ratio
                    p3_x = p0_x + (p1_y - p0_y) * len_ratio;
                    p3_y = p0_y + (p0_x - p1_x) * len_ratio;
                    p2_x = p1_x + (p1_y - p0_y) * len_ratio;
                    p2_y = p1_y + (p0_x - p1_x) * len_ratio;
                # rectify
                # aim at in_99_0614  padding version
                # 200 long-side, 90 short-side
                long_side = 150 # avm_05
                short_side = 75 #
                # long_side = 200 # in99_0614
                # short_side = 90 #
                if pow((p0_x - p3_x) ** 2 + (p0_y - p3_y) ** 2, 0.5) > long_side - 20:
                    p3_x = p0_x + long_side * vec_sep[0]
                    p3_y = p0_y + long_side * vec_sep[1]
                if pow((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2, 0.5) > long_side - 20:
                    p2_x = p1_x + long_side * vec_sep[0]
                    p2_y = p1_y + long_side * vec_sep[1]
                if pow((p0_x - p1_x) ** 2 + (p0_y - p1_y) ** 2, 0.5) > long_side - 20 and \
                   pow((p0_x - p3_x) ** 2 + (p0_y - p3_y) ** 2, 0.5) > short_side - 10:
                    p3_x = p0_x + short_side * vec_sep[0]
                    p3_y = p0_y + short_side * vec_sep[1]
                if pow((p0_x - p1_x) ** 2 + (p0_y - p1_y) ** 2, 0.5) > long_side - 20 and \
                   pow((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2, 0.5) > short_side - 10:
                    p2_x = p1_x + short_side * vec_sep[0]
                    p2_y = p1_y + short_side * vec_sep[1]
                # rectify
                p3_x, p3_y = rectify_out_of_boundary(p3_x, p3_y,
                                                     p0_x, p0_y,
                                                     width, height)
                p2_x, p2_y = rectify_out_of_boundary(p2_x, p2_y,
                                                     p1_x, p1_y,
                                                     width, height)
                p2_x = int(p2_x)
                p2_y = int(p2_y)
                p3_x = int(p3_x)
                p3_y = int(p3_y)
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
        cv2.putText(image, f'({W},{H})', (5, 15),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        # 画车位掩码图，分三步
        # 第一步：画未被占概率，画是否可用提示
        # avail = 'N' if isOccupied > 0.9 else 'Y'
        # cv2.putText(image, avail,
        #             ((p0_x + p1_x) // 2 + 10, (p0_y + p1_y) // 2 + 10),
        #             cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        area = np.array([[p0_x, p0_y], [p1_x, p1_y],
                        [p2_x, p2_y], [p3_x, p3_y]])
        if available > 0.7:
            color = (0, 255, 0)  # 置信度颜色
            cv2.putText(image,
                        f'{available:.3f}',  # 转换为"未被占用"置信度
                        ((p0_x + p2_x) // 2, (p0_y + p2_y) // 2 + 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, color)
        # 第二步：判断，如果可用，才画掩码图，否则全零
        mask = np.zeros_like(image)
        if available > 0.7:
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


def rotate_vector(vector, angle_degree=90):
    """Rotate a vector with given angle in degree."""
    angle_rad = math.pi * angle_degree / 180
    xval = vector[0] * math.cos(angle_rad) + \
           vector[1] * math.sin(angle_rad)
    yval = -vector[0] * math.sin(angle_rad) + \
           vector[1] * math.cos(angle_rad)
    return xval, yval

