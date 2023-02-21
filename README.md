Artificial neural networks have unlocked countless applications in medical image computing. Thanks to artificial intelligence development over the past few years, it is
possible for machines to visually detect in real time almost anything. This has proven
very helpful for vision based minimally invasive surgery, like Laparoscopy.
The objective of this thesis is to implement a one-stage object detection network and
train it to detect laparoscopic instruments in real time video images. Then, adapt the
network to be able to predict rotating bounding boxes with arbitrary orientation in the
most optimal way.
An object detection model was created based on SSD, and was trained to detect 18
different laparoscopic tools. Then this model was modified to not only detect the 18
different objects but also, to predict the angle and exact rotated bounding box measures.
Both models shared pre-trained layers from VGG16 for feature extraction.
The first model proved to be capable of detecting laparoscopic tools with axis aligned
bounding boxes accurately and precisely. The second model proved to be capable of
more precisely fitting bounding boxes around objects, using rotated bounding boxes.
However, further refinement is necessary for it to function optimally. Nevertheless,
the model holds great potential in enhancing the performance of object detection in
laparoscopic surgeries.



NOTES:
In this folder you can find diferent versions of the model used during the thesis ( some changes must be made according if you want to use rotated bbox or not ).

The given json files are the ones to be used for the rotated bounding box approach where rotation angle is given, and rotated bbox coordinates are used.

The reformat json files are used to obtain the proper json files to use given the original object data files, in each version you can find the differences explained.

I also include all other files needed for training and using the model, TESTER is the program to use the trained model. 


