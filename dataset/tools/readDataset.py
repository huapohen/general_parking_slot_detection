import json
import os
from collections import namedtuple

import cv2
import cv2 as cv
import json
import os.path
import numpy as np

def drawCircle(img,x,y,color,text,debug):
    if debug == True:
        h,w,c = img.shape
        circle_x = min( max( int(x) , 0), w)
        circle_y = min(max(int(y), 0), h)
        cv.circle(img, (circle_x,circle_y), 4, color, 3)
        # cv.imshow("xxx",img)
        # cv.waitKey(0)
        if text !="":
            cv.putText(img,text,(int(x), int(y)),1,1,color)
    return

def drawLine(img,point_1,point_2,color,text,debug):
    if debug == True:

        cv.line(img,(int(point_1[0]),int(point_1[1])),(int(point_2[0]),int(point_2[1]) ), color,2)
        if text !="":
            cv.putText(img,text,(int(x), int(y)),2,4,color)
    return

def getLineKB(x0,y0,x1,y1):
    delta = 0.00001
    k = (y0-y1)/(x0-x1+delta)
    b= -k*x0+y0
    return k,b

def point2lineDis(x0,y0,k,b):
    dis = np.abs( (-k*x0+y0-b)/np.sqrt(1+k**2) )
    return dis

def point2pointDis(x0,y0,x1,y1):
    dis = np.sqrt( (x0-x1)**2 + (y0-y1)**2 )
    return dis

def rotatePoint(src_x,src_y,anchor_x,anchor_y,img):
    theta = np.pi/2
    delta = 0.00001
    w,h,c = img.shape
    if src_x == anchor_x:
        theta = np.pi/2
    else:
        theta = np.arctan((src_y-anchor_y)/(src_x-anchor_x))

    r = 40
    dst_x = 0
    dst_y = 0
    # 左上
    text = ""
    if src_x < anchor_x and src_y < anchor_y:
        dst_x = anchor_x + abs(r * np.sin(theta) )
        dst_y = anchor_y - abs(r * np.cos(theta) )
        # if src_x < w / 2:
        #     dst_x = anchor_x - abs(r * np.sin(theta))
        #     dst_y = anchor_y - abs(r * np.cos(theta))
        text ="LU"
    # 左下
    if src_x < anchor_x and src_y >= anchor_y:
        dst_x = anchor_x - abs(r * np.sin(theta))
        dst_y = anchor_y - abs(r * np.cos(theta))
        if src_x >= w / 2:
            dst_x = anchor_x + abs(r * np.sin(theta))
            dst_y = anchor_y + abs(r * np.cos(theta))
        text ="LD"
    # 右上
    if src_x >= anchor_x and src_y < anchor_y:
        dst_x = anchor_x + abs(r * np.sin(theta))
        dst_y = anchor_y + abs(r * np.cos(theta))
        # if src_x < w/2:
        #     dst_x = anchor_x - abs(r * np.sin(theta))
        #     dst_y = anchor_y + abs(r * np.cos(theta))
        text ="RU"
    # 右下
    if src_x >= anchor_x and src_y >= anchor_y:
        # dst_x = anchor_x + abs(r * np.sin(theta))
        # dst_y = anchor_y + abs(r * np.cos(theta))
        dst_x = anchor_x + (r * np.sin(np.pi - theta))
        dst_y = anchor_y + (r * np.cos(np.pi - theta))
        if src_x < w / 2:
            dst_x = anchor_x - abs(r * np.sin(theta))
            dst_y = anchor_y + abs(r * np.cos(theta))
        text ="RD"

    drawCircle(img,dst_x,dst_y,(0,255,0),text,True)
    return dst_x,dst_y

# use to deal with the L shape point in ps2.0(tongji)
def readTongjiLabel(name,src_path,dst_path):
    debug = True
    mode_list = ["train"]
    output_label = []

    for mode_name in mode_list:
        out_path = os.path.join(dst_path, name,mode_name,"debug\\")
        os.makedirs(out_path , exist_ok = True)
        target_path = os.path.join(src_path, name, mode_name, "label")
        print(target_path)
        label_files = os.listdir(target_path)
        modify_path = os.path.join(src_path, name, mode_name, "modify_label")
        os.makedirs(modify_path, exist_ok=True)

        for i, label_file in enumerate(label_files) :
            change_point_flag = False
            # if i % 100 != 0:
            #     continue
            img_path = os.path.join(target_path, "..//image//")
            img_name = label_file.split(".")[0]
            # if img_name != "image20160722192751_1908":
            #     continue
            img = cv2.imread(img_path+img_name+".jpg" )
            outside_points = []
            inside_points = []

            with open(os.path.join(target_path, label_file), 'r') as file:
                marks = []
                label = json.load(file)

                points = np.array(label["points"]).reshape( (-1,4))
                inside_points = points[:,0:2]
                outside_points = points[:,2:4]
                point_shape = np.array(label["shape"])
            # filter L shape point -1 invalid 0 T-shape 1 L-shape
            line_kb = []
            if len( inside_points) > 1 :
                # calculate k,b param in inside points
                line_point_member = []
                for p_i in np.arange(0,len(inside_points) ):
                    # if point_shape[p_i] != 1:
                    #     continue
                    for p_j in np.arange(p_i+1,len(inside_points) ):
                        if point_shape[p_i] != 1 and point_shape[p_j] != 1:
                            continue
                        x0 = inside_points[p_i][0]
                        y0 = inside_points[p_i][1]
                        x1 = inside_points[p_j][0]
                        y1 = inside_points[p_j][1]
                        dis = point2pointDis(x0,y0,x1,y1)
                        drawCircle(img, x0, y0, (255, 0, 0), "",debug)
                        drawCircle(img, x1, y1, (255, 0, 0), "",debug)
                        # cv.imshow("xxx", img)
                        # cv.waitKey(0)
                        w,h,c = img.shape

                        # reject the points which too close or too far
                        # if dis > w/3:
                        #     continue
                        if dis < 60:
                            continue
                        k,b = getLineKB(x0,y0,x1,y1)
                        drawLine(img,inside_points[p_i],inside_points[p_j],(0, 255,0),"",debug)
                        line_kb.append( [k,b])
                        line_point_member.append( [x0,y0,x1,y1])
                # 1. determine the outside point is on the line using (k,b)

                for p_i in np.arange(0,len(outside_points)):
                    x0 = outside_points[p_i][0]
                    y0 = outside_points[p_i][1]

                    drawCircle(img, x0, y0, (0, 0, 255), "", debug)

                    for j in np.arange(0,len(line_kb)):
                        dis = point2lineDis(x0,y0,line_kb[j][0],line_kb[j][1])
                        dis_AB = point2pointDis(x0,y0,line_point_member[j][0],line_point_member[j][1])
                        dis_AC = point2pointDis(x0, y0, line_point_member[j][2], line_point_member[j][3])
                        dis_BC = point2pointDis(line_point_member[j][0],line_point_member[j][1], line_point_member[j][2], line_point_member[j][3])
                        # drawLine(img,line_point_member[j][0:2],line_point_member[j][2:4],(0, 0,255),"",debug)
                        # cv.imshow("xxx", img)
                        # cv.waitKey(0)
                        if dis_AB > dis_BC or dis_AC > dis_BC:
                            drawLine(img, line_point_member[j][0:2], line_point_member[j][2:4], (0, 255, 0), "", debug)
                            continue
                        drawLine(img,line_point_member[j][0:2],line_point_member[j][2:4],(0, 255,0),"",debug)
                        if dis < 11:
                            # 2. the point is on the line. we need match this point to nearest L shape point
                            nearest_point = inside_points[p_i]
                            dst_x,dst_y = rotatePoint(x0, y0, nearest_point[0], nearest_point[1], img)
                            outside_points[p_i][0] = dst_x
                            outside_points[p_i][1] = dst_y
                            change_point_flag = True
                            break

            if change_point_flag == True:
                # cv2.imwrite(out_path + img_name + ".jpg", img)
                points[:, 2:4] = outside_points
                label["points"] = points.tolist()
                with open(os.path.join(modify_path, label_file), 'w') as f:
                    json.dump(label, f, indent=4)


    print("finish seoul convert")
    return 0
    
##############################################################
# this fuction use to draw the GT.
# label description:
# ------points
# points is a MXN array . each rows means two points of the seperate line.
# points[0,:2] means the point on the entrance line
# points[0,2:4] means the point on the other side of the seperate line.
# if the point values are "-1", it means this image is negetive sample.
# ------shape
# shape is the label use in ps2.0. len(points) == len(shape)
# -1 is invalid , 0 is T shape , 1 is L shape
# ------slot_type
# This info is use in seoul dataset combined whit
# "parallel":0 , "perpendicular":1 , "diagonal":2 and "not-parking-space":3
# ------dataset_name
# tongji or seoul
##############################################################
def drawLabel(name,src_path,dst_path):
    debug = True
    mode_list = ["train","test"]
    output_label = []

    for mode_name in mode_list:
        out_path = os.path.join(dst_path, name,mode_name,"debug\\")
        os.makedirs(out_path , exist_ok = True)
        target_path = os.path.join(src_path, name, mode_name, "label")
        print(target_path)
        label_files = os.listdir(target_path)
        for i, label_file in enumerate(label_files) :
            img_path = os.path.join(target_path, "..//image//")
            img_name = label_file.split(".")[0]

            img = cv2.imread(img_path+img_name+".jpg" )
            outside_points = []
            inside_points = []

            with open(os.path.join(target_path, label_file), 'r') as file:
                marks = []
                label = json.load(file)

                points = np.array(label["points"]).reshape( (-1,4))
                inside_points = points[:,0:2]
                outside_points = points[:,2:4]
                for point in inside_points:
                    drawCircle(img,point[0],point[1],(255,0,0),"",debug)
                for point in outside_points:
                    drawCircle(img, point[0], point[1], (0, 0, 255), "",debug)
                for index in np.arange(0,len(inside_points)):
                    drawLine(img,inside_points[index],outside_points[index],(0, 255,0),"",debug)
                cv2.imwrite(out_path + img_name+".jpg",img)


if __name__ == '__main__':
    src_path = "E:\\code\\AutoParkAssist\\dataset\\dst_data"
    dst_path = "E:\\code\\AutoParkAssist\\dataset\\dst_data"
    dataset_name = ["tongji","seoul"]

    for name in dataset_name:
        drawLabel(name,src_path,dst_path)

