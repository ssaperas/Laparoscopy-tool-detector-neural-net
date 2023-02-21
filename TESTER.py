from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from math import sqrt
import cv2 
from testrectangle import plot_rotated_boxes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Decide whether to make 'normal' object detection with bounding boxes or to make object detection with rotated/oriented bounding boxes.
use_rotated_boxes = True

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

#/data/home/sas44100/images/Frames/Videos_25fps/0005_201/10000/f_10003.png
img_path ="/data/home/sas44100/images/Frames/Videos_25fps/0007_217/11000/f_11001.png"
original_image = Image.open(img_path, mode='r')
original_image = original_image.convert('RGB')

min_score=0.2
max_overlap=0.2
top_k=200
  

plt.imshow(Image.open(img_path))
plt.axis(False)
plt.margins(0,0)

# Transform
image = normalize(to_tensor(resize(original_image)))

# Move to default device
image = image.to(device)

# Forward prop.

# Forward prop.
if(use_rotated_boxes == True):
    predicted_locs, predicted_scores, predicted_angles = model(image.unsqueeze(0))  # (N, 8732, 5), (N, 8732, n_classes)
    predicted_angles = torch.sigmoid(predicted_angles) # to get range between [0:1]
    predicted_angles *= math.pi  # -> range between [0:3.1415]


else:
    predicted_locs, predicted_scores = model(image.unsqueeze(0))  # (N, 8732, 4), (N, 8732, n_classes)

# Detect objects in SSD output
print('pred angles shape: ', predicted_angles.shape)
det_boxes, det_labels, det_scores, det_angles = model.detect_objects(predicted_locs, predicted_scores, predicted_angles, min_score=min_score,
                                                            max_overlap=max_overlap, top_k=top_k, isTrain=False)
listPred_locs=""

print("det_boxes",det_boxes,"det_angles", det_angles)
print('output det boxes from function', len(det_boxes[0]))
# Move detections to the CPU
det_boxes = det_boxes[0].to('cpu')

original_dims = torch.FloatTensor(
    [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
det_boxes = det_boxes * original_dims

# Decode class integer labels
det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

# If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
# if det_labels == ['background']:
#     # Just return original image
#     return original_image

# Annotate
# annotated_image = original_image
# draw = ImageDraw.Draw(annotated_image)
font = ImageFont.truetype("/data/home/sas44100/g_laufwerk/tutorial/first_implementation/object_detection/a_PyTorch_Tutorial_to_Object_Detection_master/calibril.ttf", 15)
#for the image plot
plt.imshow(Image.open(img_path))
plt.axis(False)
plt.margins(0,0)
# print("len det classes",len(det_boxes[0]), len(det_angles[0]), len(det_labels), len(det_scores))
# Suppress specific classes, if needed
# print("det_angles",det_angles)
# print('det boxes size 0: ', det_boxes.size(0))

for i in det_boxes:
    print('box coordinates: ', i)

for i in range(det_boxes.size(0)):
    # if suppress is not None:
    #     if det_labels[i] in suppress:
    #         continue


    ### Shift boxes to left upper.

    # Boxes
    box_location = det_boxes[i].tolist()
    print('box location: ', box_location)
    # print("det_angle[i]",len(det_boxes), len(det_angles), det_boxes, det_angles, det_angles[0][1])
    ### Or here: Shift boxes to left upper.
    #box_location[i] =(x,y,witdth,height)
    #print("box_location[0]",box_location[0])
    box=box_location
    rotation = det_angles[0][i] * 180 / math.pi
    # print('box location2: ', box_location)
    print('angles= ',rotation)
    ### Rotate boxes.
    center=(box[0],box[1])
    #center = (400, 500)
    width = (box[2]-box[0])/2
    height = (box[3]-box[1])/2
    degrees = rotation
    color= label_color_map[det_labels[i]]
    
    
    #center=(box[0]-((box[2]-box[0])/2),box[1]-((box[3]-box[1])/2))
    boxs=plot_rotated_boxes(center,width,height,degrees,color)
    print("boxs",boxs)
savepath= '/data/home/sas44100/disk/image_results/f_1001_6.0.png'
plt.savefig(savepath)
plt.savefig(savepath, bbox_inches='tight',pad_inches = 0 )
    # annotated_image.show()
