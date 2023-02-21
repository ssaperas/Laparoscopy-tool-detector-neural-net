#####Code to calculate the jaccard index between segmentation data and bounding boxes

from torchvision import transforms

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from math import sqrt
import cv2 

import numpy as np
from shapely.geometry import Polygon





# box_1 = [[511, 41], [577, 41], [577, 76], [511, 76]]
# box_2 = [[544, 59], [610, 59], [610, 94], [544, 94], [100,80]]

# print(calculate_iou(box_1, box_2))

# Load model checkpoint
data= {
    "boxes": [
      [
        551.5242304573303,
        677.9886709138865,
        698.1461387793815,
        1099.00410514498
      ],
      [
        229.66333703015817,
        788.4253281945322,
        543.4392130345531,
        954.4499488254608
      ],
      [
        1211.9795359694403,
        633.1569957705224,
        1346.965320406173,
        794.9918326102409
      ],
      [
        1147.5564901526911,
        882.2010688666132,
        1557.8838087412296,
        1095.8798287173638
      ]
    ],
    "labels": [
      12,
      13,
      14,
      15
    ],
    "image_id": 10000,
    "angle": [
      -1.917050087886599,
      2.7614397700464623,
      -2.9724736165633936,
      1.6357279713083417
    ],
    "segmentation": [
      [
        [
          328.63,
          680.47,
          336.2,
          683.17,
          343.77,
          687.5,
          350.8,
          692.37,
          357.84,
          697.24,
          362.7,
          704.27,
          366.49,
          711.84,
          370.82,
          719.41,
          374.06,
          726.98,
          376.22,
          735.1,
          378.93,
          743.75,
          381.09,
          751.87,
          382.72,
          759.98,
          383.8,
          768.09,
          383.8,
          776.75,
          383.8,
          785.4,
          383.8,
          794.05,
          382.72,
          802.71,
          381.09,
          810.82,
          378.39,
          818.39,
          449.24,
          790.81,
          555.8,
          752.41,
          638.55,
          721.04,
          657.48,
          710.22,
          650.45,
          704.81,
          643.42,
          700.48,
          635.31,
          697.78,
          627.19,
          697.24,
          619.08,
          695.61,
          610.97,
          694.53,
          602.31,
          693.99,
          594.74,
          689.12,
          692.1,
          642.07,
          741.32,
          608.53,
          748.89,
          604.75,
          745.65,
          596.63,
          735.37,
          586.9,
          716.44,
          580.41,
          708.32,
          577.7,
          700.75,
          573.92,
          693.18,
          570.13,
          684.53,
          570.13,
          675.33,
          570.13,
          666.68,
          570.13,
          658.02,
          570.13,
          649.91,
          571.75,
          633.68,
          572.83,
          514.69,
          617.19,
          429.23,
          650.72
        ]
      ],
      [
        [
          58.19,
          767.55,
          164.2,
          731.85,
          272.92,
          696.7,
          328.63,
          680.47,
          336.2,
          683.17,
          343.77,
          687.5,
          350.8,
          692.37,
          357.84,
          697.24,
          362.7,
          704.27,
          366.49,
          711.84,
          370.82,
          719.41,
          374.06,
          726.98,
          376.22,
          735.1,
          378.93,
          743.75,
          381.09,
          751.87,
          382.72,
          759.98,
          383.8,
          768.09,
          383.8,
          776.75,
          383.8,
          785.4,
          383.8,
          794.05,
          382.72,
          802.71,
          381.09,
          810.82,
          378.39,
          818.39,
          281.57,
          854.09,
          177.72,
          895.74,
          119.85,
          921.7,
          108.49,
          904.93,
          93.35,
          869.78,
          79.82,
          836.24,
          65.76,
          799.46
        ]
      ],
      [
        [
          1131.83,
          701.56,
          1148.6,
          642.07,
          1165.37,
          580.95,
          1172.4,
          575.54,
          1179.43,
          571.21,
          1185.92,
          565.8,
          1192.95,
          561.48,
          1201.07,
          559.31,
          1208.64,
          555.53,
          1216.75,
          553.36,
          1225.4,
          553.36,
          1233.52,
          554.99,
          1241.09,
          558.23,
          1248.12,
          563.64,
          1254.07,
          569.59,
          1258.94,
          576.62,
          1263.81,
          583.65,
          1269.22,
          590.14,
          1273.0,
          597.71,
          1276.25,
          605.29,
          1276.25,
          613.94,
          1269.76,
          665.32,
          1264.89,
          724.28,
          1260.56,
          717.25,
          1256.23,
          709.68,
          1251.37,
          702.65,
          1245.42,
          696.15,
          1238.39,
          691.29,
          1230.81,
          687.5,
          1222.7,
          684.26,
          1215.13,
          681.01,
          1207.01,
          680.47,
          1198.36,
          679.93,
          1190.25,
          679.39,
          1181.59,
          679.39,
          1172.94,
          679.39,
          1165.37,
          682.63,
          1158.34,
          687.5,
          1150.76,
          690.75,
          1143.73,
          695.07
        ]
      ],
      [
        [
          1027.63,
          1080.0,
          1058.95,
          969.34,
          1097.08,
          824.65,
          1131.83,
          701.56,
          1143.73,
          695.07,
          1150.76,
          690.75,
          1158.34,
          687.5,
          1165.37,
          682.63,
          1172.94,
          679.39,
          1181.59,
          679.39,
          1190.25,
          679.39,
          1198.36,
          679.93,
          1207.01,
          680.47,
          1215.13,
          681.01,
          1222.7,
          684.26,
          1230.81,
          687.5,
          1238.39,
          691.29,
          1245.42,
          696.15,
          1251.37,
          702.65,
          1256.23,
          709.68,
          1260.56,
          717.25,
          1264.89,
          724.28,
          1257.47,
          815.68,
          1248.49,
          917.74,
          1241.76,
          1043.36,
          1241.76,
          1080.0
        ]
      ]
    ],
    "upright boxes": [
      [
        328.63,
        570.13,
        420.26,
        248.26
      ],
      [
        58.19,
        680.47,
        325.61,
        241.23
      ],
      [
        1131.83,
        553.36,
        144.42,
        170.92
      ],
      [
        1027.63,
        679.39,
        237.26,
        400.61
      ]
    ]
  }
def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    print("box1")
    poly_2 = Polygon(box_2)
    print("box2")
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

def average(numbers):
    avg = sum(numbers) / len(numbers)
    return avg

def calculate_jaccard_index(box, segmentation):
    # Convert segmentation data to binary image
    segmentation = np.array(segmentation)
    mask = np.zeros_like(segmentation)
    mask[segmentation > 0] = 1

    # Crop mask by bounding box
    x, y, w, h = box
    x, y, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    mask = mask[y:y2, x:x2]

    # Calculate Jaccard index
    intersection = np.sum(mask == 1)
    union = np.sum(mask >= 0)
    jaccard_index = intersection / union
    return jaccard_index

def separate_lists(seg):
    x = seg[::2]
    y = seg[1::2]
    return x, y

def reorder(data):
    #data = ["x1", "y1", "x2", "y2", "x3", "y3"]
    result = [[data[i], data[i+1]] for i in range(0, len(data), 2)]
    return result

#data = ["x1", "y1", "x2", "y2", "x3", "y3","x3", "y3"]
seg=data['segmentation']
#print(seg)
jaccardidxlistu=[]
jaccardidxlistr=[]
for i in range(len(data['segmentation'])):
  #print(len(data["segmentation"]), len(data["upright boxes"]))

  segm=data['segmentation'][i][0]
    
  box= data["boxes"][i]

  center=(box[0],box[1])
  width = (box[2]-box[0])/2
  height = (box[3]-box[1])/2
  x=box[0]-width/2
  y= box[1]-height/2
  boxs =[[x,y],[x+box[2],y],[x+box[2],y+box[3]], [x,y+box[3]]]
 #for non rotated objects
  box= data["upright boxes"][i]
  boxx =[[box[0], box[1]],[box[0]+box[2],box[1]],[box[0]+box[2],box[1]+box[3]], [box[0],box[1]+box[3]]]
  print("box: ",boxx)
  
  #for b in a:
  # print(a, "#######")
  
  segmentation=reorder(segm)
  # print("segmentation:",segmentation)
  # print("box: ",boxx)
  # print(calculate_iou(boxx, segmentation))
  jaccard_indexa = calculate_iou(boxx, segmentation)
  jaccardidxlistu.append(jaccard_indexa)
  jaccard_indexb = calculate_iou(boxs, segmentation)
  jaccardidxlistr.append(jaccard_indexb)

print("non rotated boxes: ",jaccardidxlistu,"average=", average(jaccardidxlistu))
print("rotated boxes: ",jaccardidxlistr,"average=", average(jaccardidxlistr))
    #print("######",segmentation)

# Example usage
"""
box = [10, 20, 30, 40] # x, y, w, h
segmentation = [[0, 0, 1, 1], [1, 1, 1, 0]] #format segmentation???
jaccard_index = calculate_jaccard_index(box, segmentation)
print(jaccard_index)

"""
# Data to be used




