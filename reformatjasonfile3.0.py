'''
this code redistributes or data json file to fit the object detection example SDD300
also (A, B, A+x, B+y)
NEW: calculates Rotated Bounding box and the rotation angle, leaves the not rotated "boxes"
but instead adds angle in new "angle" data category

RESULTING 




'''

import json
from utils import pointmapintocoord
from utils import bboxangle
from MinimumBoundingBox import MinimumBoundingBox

f = open('/data/home/sas44100/g_laufwerk/tutorial/first_implementation/object_detection/data/TRAIN_video_5_18_classes.json')
data = json.load(f)
idlist=[]
reformat=[]
allidlist=[]
#print("img_7 has 1409 images")
print("len_llista_completa_original=",len(data['annotations']))
for i in data['annotations']:
    #allidlist.append(i["image_id"])
    if i["image_id"] not in idlist:
        idlist.append(i["image_id"])
#print(idlist)   
print("num_img_id_original=",len(idlist))
#print("num_img_id_original=",len(allidlist))
for e in idlist:
    #print("e=", e)
    reformat.append({"boxes":[],"labels":[],"image_id":e-1,"angle":[]})
    #e= e-1
#print(data['annotations'][0]["image_id"])
for d in range(len(reformat)):
    for c in range(len(data['annotations'])):
        
        if data['annotations'][c]["image_id"] == (reformat[d]["image_id"]+1):
            #print("yes")
            #segmentation list into ((x,y),(x1,y1)...)format
            segmentation=data['annotations'][c]["segmentation"]
            # print('segmentation',segmentation[0])
            coordenates=pointmapintocoord(segmentation[0])
            #calculate rotated bounding box and angleand adds it to "angle" in the new json file
            bounding_box = MinimumBoundingBox(coordenates)
            rotatedbbox=bounding_box.corner_points
            #print('rotatedbbox',rotatedbbox)
            
            
            #a={(1363.4172021536901, 509.9278446026319), (1253.7709776420506, 871.7756846800221), (1293.5210365785915, 502.4059062543184), (1323.6671432171493, 879.2976230283357)}
            #angle=bboxangle(a)
            #print('angle',angle)
            rotatedangle=bboxangle(rotatedbbox)
            reformat[d]["angle"].append(rotatedangle)
            reformat[d]["boxes"].append(data['annotations'][c]["bbox"])
            reformat[d]["labels"].append(data['annotations'][c]["category_id"])
            
            #print(reformat)
            #print(reformat[d]["boxes"])
            #print("d[boxes]=", d["boxes"]) #it adds the boxes but you can't see them in reformat

        


#(A, B, A+x, B+y) operation
for d in range(len(reformat)):
    for a in reformat[d]["boxes"]:
        a[2]=a[2]+a[0]
        a[3]=a[3]+a[1]
print("len(idlist)_final",len(idlist))
print("len_reformat_final", len(reformat)  )   
idlist= [x - 1 for x in idlist] #delete -1 to every item
print('idlist=',idlist) 
data = reformat
with open('new_reformated_file.json', 'w') as f:
    json.dump(data, f, indent=2)
    print("The json file is created") 

f.close()


'''
Deletes image paths from image_file if they don't appear in object_file
(deletes extra image paths and tells you the extra objects)
'''
path='/data/home/sas44100/g_laufwerk/tutorial/first_implementation/object_detection/data/TRAIN_5_images(1st).json'

g = open(path)
dataimg = json.load(g)
print("len(dataimg)inici=",len(dataimg))
#dataimg= dataimg[:20]
print("len idlist",len(idlist))
missing_image=idlist
for k in missing_image[:]:

    for l in dataimg[:]:
        #print(l)
        mk1= l.find('f_')+2
        mk2= l.find('.png',mk1)
        num_img=int(l[mk1:mk2])
        if num_img == k:
            missing_image.remove(k) 
        #num_img=int((l.split("f_"))[1].split(".png")[0])
    # print("X-num_img=", num_img," // l= ",l)
        if num_img not in idlist:
           # print(num_img,"not in idlist //",l,"removed" )
            dataimg.remove(l)
print("missing_image=", missing_image)#delete these from the object json file manually, they are extra data with no matching frames.
print("len(dataimg)final=",len(dataimg))
with open('TRAIN_5_images(NEW).json', 'w') as g:
    json.dump(dataimg, g, indent=2)
    print("idlist=",idlist)
    print("The json file img is created") 

g.close()

