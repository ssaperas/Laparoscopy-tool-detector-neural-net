'''
this code redistributes or data json file to fit the object detection example SDD300
     also (A, B, A+x, B+y)
     '''

import json
from utils import pointmapintocoord

f = open('/data/home/sas44100/g_laufwerk/tutorial/first_implementation/object_detection/data/example_5_objects.json')
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
    reformat.append({"boxes":[],"labels":[],"image_id":e-1})
    #e= e-1
#print(data['annotations'][0]["image_id"])
for d in range(len(reformat)):
    for c in range(len(data['annotations'])):
        
        if data['annotations'][c]["image_id"] == (reformat[d]["image_id"]+1):
            #print("yes")

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
idlist= [x - 1 for x in idlist]
print(idlist) 
data = reformat
with open('new_reformated_file.json', 'w') as f:
    json.dump(data, f, indent=2)
    print("The json file is created") 

f.close()
'''
Deletes image paths from image_file if they don't appear in object_file
(deletes extra image paths and tells you the extra objects)
'''
g = open('/data/home/sas44100/g_laufwerk/tutorial/first_implementation/object_detection/data/TRAIN_5_images(old).json')
dataimg = json.load(g)
print("len(dataimg)inici=",len(dataimg))
#dataimg= dataimg[:20]
#print(dataimg)
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
            print(num_img,"not in idlist //",l,"removed" )
            dataimg.remove(l)
print("missing_image=", missing_image)#delete these from the object json file manually, they are extra data with no matching frames.
print("len(dataimg)final=",len(dataimg))
with open('TRAIN_5_images(NEW).json', 'w') as g:
    json.dump(dataimg, g, indent=2)
    print("The json file img is created") 

g.close()