import time
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasetsSergi import PascalVOCDataset
from utils import *
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
"""
To train only for object detection use the following dataset files
        
TRAIN_5_objects(2).json-----TRAIN_5_images(1st).json
TEST_7_objects(2).json-----TEST_7_images(1st).json

To train for rotated bounding boxes use the following dataset files     
TRAIN_5_objects(3).json-----TRAIN_5_images(1st).json
TEST_7_objects(3).json-----TEST_7_images(1st).json

COPY THE CONTENTS (for train run)to:
TRAIN_5_objects.json-----TRAIN_5_images.json
TEST_7_objects.json-----TEST_7_images.json



"""

writer = SummaryWriter('/data/home/sas44100/g_laufwerk/tutorial/first_implementation/object_detection/a_PyTorch_Tutorial_to_Object_Detection_master/tensorboard_runs/4/')

# Data parameters
data_folder = '/data/home/sas44100/g_laufwerk/tutorial/first_implementation/object_detection/data'  # folder with data files
keep_difficult = False  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters

checkpoint =None#'checkpoint_ssd300_4.0.pth.tar' #'/data/home/sas44100/g_laufwerk/tutorial/first_implementation/object_detection/a_PyTorch_Tutorial_to_Object_Detection_master/checkpoint'#None  # path to model checkpoint, None if none
batch_size = 18  # batch size
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-4  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
epochs = 200


torch.autograd.set_detect_anomaly(True)
cudnn.benchmark = True

# Decide whether to make 'normal' object detection with bounding boxes or to make object detection with rotated/oriented bounding boxes.
use_rotated_boxes = False

def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes, use_rotated_boxes=use_rotated_boxes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        # optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': lr}, {'params': not_biases}],
        #                             lr=lr, momentum=momentum, weight_decay=weight_decay)

        optimizer = torch.optim.Adam(params=[{'params': biases, 'lr': lr}, {'params': not_biases}],
                                    lr=lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train_5' )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here


    val_dataset = PascalVOCDataset(data_folder,
                                     split='test_7' )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                               collate_fn=val_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    # epochs = iterations // (len(train_dataset) // batch_size)
   # decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]
    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        # if epoch in decay_lr_at:
        #     adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              isTrain=True)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer) #saves checkpoint every epoch

        # Validation / Eval.
        with torch.no_grad():
            train(train_loader=val_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                epoch=epoch,
                isTrain=False)


def train(train_loader, model, criterion, optimizer, epoch, isTrain):
    
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    if(isTrain):
        model.train()  # training mode enables dropout
    else:
        model.eval()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()
    angle_losses = AverageMeter()
    conf_losses = AverageMeter()
    loc_losses = AverageMeter()
    
 # loss
    # mAP = AverageMeter() # mAP

    start = time.time()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader), pct_start=0.05)
    
    metric = MeanAveragePrecision()    # map_metric.update(preds, target)

    # Batches = Iterations.
   

    #for i, (images, boxes, labels, angles) in enumerate(tqdm(train_loader)):
    for i, (images, boxes, labels) in enumerate(tqdm(train_loader)):
        
        data_time.update(time.time() - start)

        

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        if(use_rotated_boxes == True):
          angles = [e.to(device) for e in angles]

        # Forward prop.
        if(use_rotated_boxes == True):
            predicted_locs, predicted_scores, predicted_angles = model(images)  # (N, 8732, 5), (N, 8732, n_classes)
            predicted_classes = torch.argmax(predicted_scores, 2)

            pred_angles = torch.sigmoid(predicted_angles)
            pred_angles *= math.pi
            
            predicted_angles_detached = predicted_angles.detach().clone()
            # print('predicted angles: ', predicted_angles_detached)
            predicted_angles_detached = torch.sigmoid(predicted_angles_detached) # to get range between [0:1]
            predicted_angles_detached *= math.pi  # -> range between [0:3.1415]

            

            # print('boxes shape: ', boxes.shape)
            # print('shapes: ', predicted_locs.shape, predicted_scores.shape, predicted_angles.shape)
            # print('shape after argmax: ', predicted_classes.shape)
        else:
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)



        # TODO: Maybe run detect_objects for each batch element. Index batch element, and add first dimension by unsqueeze.
        

        # Detect objects in SSD output
        min_score = 0.2
        max_overlap = 0.2
        top_k = 200
        # print('pr
        # ed angles shape: ', pred_angles.shape)
        # det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, pred_angles, min_score=min_score,
        #                                                      max_overlap=max_overlap, top_k=top_k, isTrain=use_rotated_boxes)
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k, isTrain=use_rotated_boxes)
        
        
        target = []
        preds = []
        
        #Tlist = torch.Tensor().to(device)#tensor with list of all boxes in a batch (FIX ERROR WITH DEVICE)
        #det_boxes/det_labels... are lists of 8
        ####PREDS#####
        for img_num in range(len(det_boxes)):#iterates for 8 batches
            
            img_boxes = det_boxes[img_num]
            img_labels = det_labels[img_num]
            img_scores = det_scores[img_num]
            Tlist = torch.Tensor().to(device)
            for i in img_boxes: #in each batch boxes are stored as a list of tensors We want a list in a single tensor
                
               Tlist=torch.cat((Tlist,i),-1)
          
            D = dict(boxes=img_boxes, labels= img_labels, scores = img_scores)# 1 dict for each batch
            preds.append(D)
           # print("len: ",len(img_labels),len(img_boxes),len(img_boxes),img_boxes)#, "preds: ",preds )
        ####TARGET####
        for img_num in range(len(boxes)):#shape of boxes is 8, and in each batch boxes are stored as a list in a tensor
            img_boxes = boxes[img_num]
            img_labels = labels[img_num]
            B = dict(boxes=img_boxes, labels= img_labels)# 1 dict for each batch
            target.append(B)
      
        #print(target)
        
        # for img_num in range(len(boxes)):#iterates for 8 batches
            
        #     img_boxes = boxes[img_num]
        #     img_labels = labels[img_num]
        #     print("img-box", img_boxes)
            
        #     for i in img_boxes:
                
        #        Tlist=torch.cat((Tlist,i),0)
          
        #     D = dict(boxes=Tlist, labels= img_labels)# 1 dict for each batch
        #     target.append(D)
        # print("boxlist",Tlist)
 


        # img_top_boxes = img_boxes[0]
        # print(img_top_boxes)
        # print('zero det box size: ', zero_det_box.size(0))

        # for i in range(zero_det_box.size(0)):
        #     box_location = det_boxes[i].tolist()

            # print('box location: ', box_location)
            # print('len box locations: ', len(box_location))



        # print('det boxes: ', len(det_boxes))
        # print('det boxes 0 : ', len(det_boxes[0]))
        # print('det boxes 0 0 : ', len(zero_det_box[0]))
        # print('det boxes 0 0 : ', zero_det_box[0])
        # print(len(det_labels))
        # print(len(det_scores))
        # print(custom_boxes.shape)
        # print(custom_labels.shape)

        # det_boxes = torch.stack(det_boxes)
        # det_labels = torch.stack(det_labels)
        # det_scores = torch.stack(det_scores)

        # print('### Modified: ')
        # print(det_boxes.shape)
        # print(det_labels.shape)
        # print(det_scores.shape)
        # print(custom_boxes.shape)
        # print(custom_labels.shape)

        if(use_rotated_boxes == True):
             #added this "loc_loss, angle_loss"   
            loss,loc_loss, angle_loss, conf_loss = criterion(predicted_locs, predicted_scores, predicted_angles, boxes, labels, angles)  # scalar
            #print("lengths: ", len(predicted_locs),len(boxes),len(predicted_angles),len(angles),len(predicted_scores))
            # APs, mAP = calculate_mAP_old(det_boxes, det_labels, det_scores, true_boxes, true_labels, device)
            #APs, mAP = calculate_mAP(det_boxes, det_labels , det_scores, det_angles, true_boxes=boxes, true_labels=labels, true_angles=angles, device=device)
           # print("loss: ", loss)
        else:
           #loss = criterion(predicted_locs, predicted_scores, None, boxes, labels)
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
            # APs, mAP = calculate_mAP_old(det_boxes=predicted_locs, det_labels, det_scores=predicted_scores, true_boxes=boxes, true_labels=labels, device)
       
        

        metric.update(preds, target)       
        # if(isTrain):
        #     metric.update(preds, target)    # print('map metric: ', map_metric.compute())
        # else:
        #     metric = MeanAveragePrecision()
        #     metric.update(preds, target)
        #     meanAP = metric.compute()
        #     del metric

       # print("#######____mAP____######= ",mAP_value, "EPOCH: ", epoch)

        #NOTES: implement mean average precision, use predicted_locs,scores,angles above to compare them with boxes, labels...
        # print(loss)
        # print("mAP= ",mAP)
        if(isTrain):
            # Backward prop.
            optimizer.zero_grad()
            loss.backward() #is it updating the angles aswell?
            

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        if(isTrain):
            # Update model
            optimizer.step()
            

        scheduler.step()

        losses.update(loss.item(), images.size(0))
        loc_losses.update(loc_loss.item(), images.size(0))
        conf_losses.update(conf_loss.item(), images.size(0))
        angle_losses.update(angle_loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

       
    epoch_mAP = metric.compute()
    mAP_value = epoch_mAP['map']
    epoch_loss = losses.avg
    loc_l = loc_losses.avg
    conf_l = conf_losses.avg
    angle_l = angle_losses.avg
    # epoch_mAP = map_metric.compute()
#UNCOMENT
    # epoch_mAP = metric.compute()
    # pprint(epoch_mAP)

    # print('Epoch mAP: ', epoch_mAP)

    
    #print('Epoch: ', epoch)

    # if(isTrain):
    #     print('Train Loss: ', epoch_loss)
    #     writer.add_scalar('Train/Loss', epoch_loss, epoch)

    # else:
    #     print('Validation Loss: ', epoch_loss)
    #     writer.add_scalar('Validation/Loss', epoch_loss, epoch)

    if(isTrain):
       # print('Train Loss: ', epoch_loss,'loc_loss: ',loc_loss,'angle_loss: ',angle_loss)
        writer.add_scalar('Train Loss', epoch_loss, epoch)
        writer.add_scalar('train_loc_loss: ',loc_l,epoch)#like this?
        writer.add_scalar('train_angle_loss: ',angle_l,epoch)#like this?
        writer.add_scalar('train_conf_loss: ',conf_l,epoch)#like this?
        writer.add_scalar('Train mAP', mAP_value, epoch)
       


    else:
        #print('Validation Loss: ', epoch_loss)
        writer.add_scalar('Validation Loss', epoch_loss, epoch)
        writer.add_scalar('Validation_loc_loss: ',loc_l,epoch)#like this?
        writer.add_scalar('Validation_angle_loss: ',angle_l,epoch)#like this?
        writer.add_scalar('Validation_conf_loss: ',conf_l,epoch)#like this?
        writer.add_scalar('Validation mAP', mAP_value, epoch)


    del metric, mAP_value, epoch_mAP, loc_l, angle_l, conf_l, loc_losses, angle_losses, conf_losses, predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()
