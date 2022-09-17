import os
import cv2
import json
import pdb
import math
import numpy as np
from collections import OrderedDict


def boundary_check(centralied_marks):
    """Check situation that marking point appears too near to border."""
    for mark in centralied_marks:
        # 每一条停车线单独遍历，若角点距离边缘过近则不使用
        if mark[0] < 40 or mark[0] > 560 or mark[1] < 40 or mark[1] > 560:
            return False
    return True


def tongji_data_process(data_path, output_path, data_type="train"):
    assert data_type in ["train", "test"]
    output_path = os.path.join(output_path, data_type)
    output_path_image = os.path.join(output_path, "image")
    output_path_label = os.path.join(output_path, "label")
    output_path_list = os.path.join(output_path, "img_list.txt")
    if not os.path.exists(output_path_image):
        os.makedirs(output_path_image)
    if not os.path.exists(output_path_label):
        os.makedirs(output_path_label)

    data_path = os.path.join(data_path, data_type)
    list_path = os.path.join(data_path, "img_list.txt")
    f_list = open(output_path_list, "w")
    with open(list_path, "r") as f_label:
        for line in f_label.readlines():

            # get data path
            data_name = os.path.splitext(os.path.split(line)[-1][:-1])[0]
            json_name = data_name + ".json"
            img_name = data_name + ".jpg"
            label_path = os.path.join(
                os.path.join(data_path, "label"), json_name)
            img_path = os.path.join(os.path.join(
                data_path, "image"), img_name)

            # read data
            with open(label_path, "r") as f:
                label = json.load(f)
            img = cv2.imread(img_path)

            if not boundary_check(label['points']):
                continue

            # process data
            h, w, c = img.shape
            margin = 40

            # cut image
            selec_range = [[margin, w//3+margin], [2*w//3-margin, w-margin]]
            img_slice_left = img[:, selec_range[0][0]:selec_range[0][1], :]
            img_slice_right = img[:, selec_range[1][0]:selec_range[1][1], :]
            img_slice_left = np.flip(img_slice_left, axis=1).copy()

            # re-assign labels
            points_in_left = []
            points_in_right = []
            shape_in_left = []
            shape_in_right = []
            direction_in_left = []
            direction_in_right = []
            for k, point in enumerate(label['points']):

                if selec_range[0][0] < point[0] < selec_range[0][1]:
                    point[0] -= selec_range[0][0]
                    point[2] -= selec_range[0][0]
                    point[0] = w//3 - point[0]
                    point[2] = w//3 - point[2]
                    direction = math.atan2(
                        point[3] - point[1], point[2] - point[0])

                    points_in_left.append(point)
                    shape_in_left.append(label['shape'][k])
                    direction_in_left.append(direction)
                    continue
                if selec_range[1][0] < point[0] < selec_range[1][1]:
                    point[0] -= selec_range[1][0]
                    point[2] -= selec_range[1][0]
                    direction = math.atan2(
                        point[3] - point[1], point[2] - point[0])
                    points_in_right.append(point)
                    shape_in_right.append(label['shape'][k])
                    direction_in_right.append(direction)
                    continue

            # save left image and label
            save_img_path = os.path.join(
                output_path_image, data_name+"_left.jpg")
            save_label_path = os.path.join(
                output_path_label, data_name+"_left.json")
            f_list.write(data_name+"_left" + " {}\n".format(len(points_in_left)))

            dicts = OrderedDict()
            dicts["points"] = points_in_left
            dicts["direction"] = direction_in_left
            dicts["shape"] = shape_in_left
            dicts["left"] = True
            dicts["dataset_name"] = "tongji"
            dicts["slot_type"] = -1

            cv2.imwrite(save_img_path, img_slice_left)
            with open(save_label_path, 'w') as f:
                json.dump(dicts, f, indent=4)

            # save right image and label
            save_img_path = os.path.join(
                output_path_image, data_name+"_right.jpg")
            save_label_path = os.path.join(
                output_path_label, data_name+"_right.json")
            f_list.write(data_name+"_right" + " {}\n".format(len(points_in_left)))

            dicts = OrderedDict()
            dicts["points"] = points_in_right
            dicts["direction"] = direction_in_right
            dicts["shape"] = shape_in_right
            dicts["left"] = False
            dicts["dataset_name"] = "tongji"
            dicts["slot_type"] = -1

            cv2.imwrite(save_img_path, img_slice_right)
            with open(save_label_path, 'w') as f:
                json.dump(dicts, f, indent=4)

    f_list.close()


def dele_repeat_item(input):
    new_list = []
    for item in input:
        if item not in new_list:
            new_list.append(item)
    return new_list


def json_modify(json_dir):
    for json_name in os.listdir(json_dir):
        json_path = os.path.join(json_dir, json_name)
        with open(json_path, "r") as f:
            json_data = json.load(f)

        json_data["points"] = dele_repeat_item(json_data["points"])
        json_data["points"] = [[int(x) for x in point]
                               for point in json_data["points"]]

        direction = [math.atan2(point[3] - point[1], point[2] - point[0])
                     for point in json_data["points"]]
        json_data["direction"] = direction
        json_data["left"] = False
        json_data["shape"] = json_data["shape"][:len(direction)]

        # 负样本判断
        if json_data["points"][0] == [-1, -1, -1, -1]:
            json_data["points"] = []
            json_data["shape"] = []
            json_data["direction"] = []

        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)

def json_modify(json_dir):
    for json_name in os.listdir(json_dir):
        json_path = os.path.join(json_dir, json_name)
        with open(json_path, "r") as f:
            json_data = json.load(f)

        json_data["points"] = dele_repeat_item(json_data["points"])
        json_data["points"] = [[int(x) for x in point]
                               for point in json_data["points"]]

        direction = [math.atan2(point[3] - point[1], point[2] - point[0])
                     for point in json_data["points"]]
        json_data["direction"] = direction
        json_data["left"] = False
        json_data["shape"] = json_data["shape"][:len(direction)]

        # 负样本判断
        if json_data["points"][0] == [-1, -1, -1, -1]:
            json_data["points"] = []
            json_data["shape"] = []
            json_data["direction"] = []

        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)

def add_line_number(list_dir):
    with open(os.path.join(list_dir, "img_list.txt"), 'w') as f_list:
        json_dir = os.path.join(list_dir, "label")
        for json_name in os.listdir(json_dir):
            json_path = os.path.join(json_dir, json_name)
            with open(json_path, "r") as f:
                json_data = json.load(f)
            line_number = len(json_data["points"])

            f_list.write(os.path.splitext(json_name)[0] + " {}\n".format(line_number))



def show_points(data_dir, data_name):
    json_path = os.path.join(os.path.join(
        data_dir, "label"), data_name+".json")
    image_path = os.path.join(os.path.join(
        data_dir, "image"), data_name+".jpg")

    with open(json_path, "r") as f:
        json_data = json.load(f)
    img = cv2.imread(image_path)

    for point in json_data["points"][-1:]:
        cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 255))
        # cv2.circle(img, (int(point[2]), int(point[3])), 5, (0, 255, 255))
    cv2.imwrite(data_name+"_drawed.jpg", img)


if __name__ == "__main__":
    # data_dir = "/data/jialanpeng"
    # tongji_path = os.path.join(data_dir, "tongji")
    # output_path = "/data/ynj/data/parking_slot_half/tongji"
    # tongji_data_process(tongji_path, output_path, data_type="train")

    list_dir = "/data/ynj/data/parking_slot_half/seoul/train"
    add_line_number(list_dir)

    # json_dir = "/data/ynj/data/parking_slot_half/seoul/train/label"
    # json_modify(json_dir)

    # data_dir = "/data/ynj/data/parking_slot_half/seoul/train"
    # data_name = "180227-143408CAM1_001068"
    # show_points(data_dir, data_name)
