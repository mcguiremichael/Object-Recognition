import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time

class YoloLoss(nn.Module):
    def __init__(self,S,B,l_coord,l_noobj):
        super(YoloLoss,self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def loss(self, a, b):
        return torch.sum( (a - b) ** 2 )

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou
    
    def get_class_prediction_loss(self, classes_pred, classes_target):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)

        Returns:
        class_loss : scalar
        """
        #diff = classes_pred - classes_target       
        ##### CODE #####
        
        class_loss = self.loss(classes_pred, classes_target.detach())
        
        return class_loss
    
    
    def get_regression_loss(self, box_pred_response, box_target_response):   
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 5)
        box_target_response : (tensor) size (-1, 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified 
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar
        
        """
        
        ##### CODE #####
        coord_loss = self.loss(box_pred_response[:,0], box_target_response[:,0].detach()) + self.loss(box_pred_response[:,1], box_target_response[:,1].detach())
        
        dim_pred = torch.sqrt(box_pred_response[:,2:4])
        dim_targ = torch.sqrt(box_target_response[:,2:4])
        dim_loss = self.loss(dim_pred[:,0], dim_targ[:,0].detach()) + self.loss(dim_pred[:,1], dim_targ[:,1].detach())
        
        return self.l_coord * (coord_loss + dim_loss)
    
    def get_contain_conf_loss(self, box_pred_response, box_target_response_iou):
        """
        Parameters:
        box_pred_response : (tensor) size ( -1 , 5)
        box_target_response_iou : (tensor) size ( -1 , 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified 
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        contain_loss : scalar
        
        """
        
        ##### CODE #####
        
        contain_loss = self.loss(box_pred_response, box_target_response_iou.detach())
        
        return contain_loss
    
    def get_no_object_loss(self, target_tensor, pred_tensor, no_object_mask):
        """
        Parameters:
        target_tensor : (tensor) size (batch_size, S , S, 30)
        pred_tensor : (tensor) size (batch_size, S , S, 30)
        no_object_mask : (tensor) size (batch_size, S , S, 30)

        Returns:
        no_object_loss : scalar

        Hints:
        1) Create a 2 tensors no_object_prediction and no_object_target which only have the 
        values which have no object. 
        2) Have another tensor no_object_prediction_mask of the same size such that 
        mask with respect to both confidences of bounding boxes set to 1. 
        3) Create 2 tensors which are extracted from no_object_prediction and no_object_target using
        the mask created above to find the loss. 
        """
        
        ##### CODE #####
        """
        no_object_prediction = pred_tensor * no_object_mask
        no_object_target = target_tensor * no_object_mask
        
        no_object_prediction_mask = torch.zeros(no_object_mask.size(), device=self.device).type(torch.ByteTensor)
        no_object_prediction_mask[:,:,:,4] = 1
        no_object_prediction_mask[:,:,:,9] = 1
        
        pred = no_object_prediction[no_object_prediction_mask]
        targ = no_object_target[no_object_prediction_mask]
        pred.requires_grad=True
        print(pred.size(), targ.size())
        result = self.loss(pred, targ.detach())
        return result
        """
        pred = pred_tensor[no_object_mask]
        targ = target_tensor[no_object_mask]
        result = self.loss(pred, targ.detach())
        return self.l_noobj * result
        
        
        
    
    
    def find_best_iou_boxes(self, box_target, box_pred):
        """
        Parameters: 
        box_target : (tensor)  size (-1, 5)
        box_pred : (tensor) size (-1, 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified 
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns: 
        box_target_iou: (tensor)
        contains_object_response_mask : (tensor)

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) Set the corresponding contains_object_response_mask of the bounding box with the max iou
        of the 2 bounding boxes of each grid cell to 1.
        3) For finding iou's use the compute_iou function
        4) Before using compute preprocess the bounding box coordinates in such a way that 
        if for a Box b the coordinates are represented by [x, y, w, h] then 
        x, y = x/S - 0.5*w, y/S - 0.5*h ; w, h = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height. 
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        5) Set the confidence of the box_target_iou of the bounding box to the maximum iou
        
        """
        
        coo_response_mask = torch.zeros((box_pred.shape[0]), device=self.device).type(torch.ByteTensor)
        box_target_iou = torch.zeros((box_pred.shape[0]), device=self.device)
        
        N = int(box_pred.shape[0] / 2)
        
        box_target = self.box_preprocess(box_target[:,:4])
        box_pred = self.box_preprocess(box_pred[:,:4])
        
        
        for i in range(N):
            iou = self.compute_iou(box_pred[i:i+2,:4], box_target[[i],:4])
            first_iou = iou[0][0]
            second_iou = iou[1][0]
            if first_iou >= second_iou:
                coo_response_mask[i] = 1
                box_target_iou[i] = first_iou
            else:
                coo_response_mask[i+1] = 1
                box_target_iou[i+1] = second_iou
        
        ##### CODE #####

        return box_target_iou, coo_response_mask
        
    
    def box_preprocess(self, boxes):
        boxes[:,0] = boxes[:,0] / self.S - 0.5 * boxes[:,2]
        boxes[:,1] = boxes[:,1] / self.S - 0.5 * boxes[:,3]
        boxes[:,2] += boxes[:,0]
        boxes[:,3] += boxes[:,1]
        return boxes
    
    
    def forward(self, pred_tensor,target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30)
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes
        
        target_tensor: (tensor) size(batchsize,S,S,30)
        
        Returns:
        Total Loss
        '''
        N = pred_tensor.size()[0]
        
        total_loss = None
        boxes_pred_tensor = pred_tensor[:,:,:,:10]
        classes_pred_tensor = pred_tensor[:,:,:,10:]
        boxes_target_tensor = target_tensor[:,:,:,:10]
        classes_target_tensor = target_tensor[:,:,:,10:]
        # Create 2 tensors contains_object_mask and no_object_mask 
        # of size (Batch_size, S, S) such that each value corresponds to if the confidence of having 
        # an object > 0 in the target tensor.
       
       
        
        contains_object_mask = ((boxes_target_tensor[:,:,:,4] + boxes_target_tensor[:,:,:,9]) > 0.0).to(self.device, dtype=torch.uint8)
        no_object_mask = ((boxes_target_tensor[:,:,:,4] + boxes_target_tensor[:,:,:,9]) == 0.0).to(self.device, dtype=torch.uint8)
        ##### CODE #####

        # Create a tensor contains_object_pred that corresponds to 
        # to all the predictions which seem to confidence > 0 for having an object
        # Split this tensor into 2 tensors :
        # 1) bounding_box_pred : Contains all the Bounding box predictions of all grid cells of all images
        # 2) classes_pred : Contains all the class predictions for each grid cell of each image
        # Hint : Use contains_object_mask
        """
        box_mask = contains_object_mask.unsqueeze(boxes_pred_tensor).expand(boxes_pred_tensor.size())
        bounding_box_pred = boxes_pred_tensor * box_mask
        class_mask = contains_object_mask.unsqueeze(3).expand(classes_pred_tensor.size())
        classes_pred = classes_pred_tensor * class_mask
        ##### CODE #####
                           
        
        # Similarly as above create 2 tensors bounding_box_target and
        # classes_target.
        
        ##### CODE #####
        contains_object_mask = ((boxes_target_tensor[:,:,:,4] + boxes_target_tensor[:,:,:,9]) > 0.0).to(self.device, dtype=torch.float32)
        no_object_mask = ((boxes_target_tensor[:,:,:,4] + boxes_target_tensor[:,:,:,9]) == 0.0).to(self.device, dtype=torch.uint8)
        box_mask = contains_object_mask.unsqueeze(3).expand(boxes_pred_tensor.size())
        bounding_box_target = boxes_target_tensor * box_mask
        class_mask = contains_object_mask.unsqueeze(3).expand(classes_pred_tensor.size())
        classes_target = classes_target_tensor * class_mask
        
        """
        
        
        
        # Compute the No object loss here
        
        ##### CODE #####
        extended_no_object_mask = torch.zeros(pred_tensor.size(), device=self.device).type(torch.ByteTensor)
        first_box_responsible = (boxes_pred_tensor[:,:,:,4] >= boxes_pred_tensor[:,:,:,9]).to(self.device, dtype=torch.uint8)
        second_box_responsible = (boxes_pred_tensor[:,:,:,9] > boxes_pred_tensor[:,:,:,4]).to(self.device, dtype=torch.uint8)
        extended_no_object_mask[:,:,:,4] = first_box_responsible * no_object_mask
        extended_no_object_mask[:,:,:,9] = second_box_responsible * no_object_mask
        no_object_loss = self.get_no_object_loss(target_tensor, pred_tensor, extended_no_object_mask)
        # Compute the iou's of all bounding boxes and the mask for which bounding box 
        # of 2 has the maximum iou the bounding boxes for each grid cell of each image.
        
        ##### CODE #####
        
        # Create 3 tensors :
        # 1) box_prediction_response - bounding box predictions for each grid cell which has the maximum iou
        # 2) box_target_response_iou - bounding box target ious for each grid cell which has the maximum iou
        # 3) box_target_response -  bounding box targets for each grid cell which has the maximum iou
        # Hint : Use contains_object_response_mask
        
        ##### CODE #####
        
        # Find the class_loss, containing object loss and regression loss
        
        ##### CODE #####
        
        
        extended_object_mask = (1 - no_object_mask).unsqueeze(3).expand(classes_target_tensor.size())
        class_prediction_loss = self.get_class_prediction_loss(classes_pred_tensor[extended_object_mask], classes_target_tensor[extended_object_mask])
        
        
        
        """
        regression_contains_object_mask = torch.zeros(boxes_pred_tensor.size()).to(self.device, dtype=torch.uint8)
        regression_contains_object_mask[:,:,:,:5] = first_box_responsible
        regression_contains_object_mask[:,:,:,5:10] = second_box_responsible
        regression_contains_object_mask = regression_contains_object_mask * contains_object_mask.unsqueeze(3).expand(regression_contains_object_mask.size())
        """
        
        pb1 = (boxes_pred_tensor[:,:,:,:5])[first_box_responsible]
        pb2 = (boxes_pred_tensor[:,:,:,5:10])[second_box_responsible]
        tb1 = (boxes_target_tensor[:,:,:,:5])[first_box_responsible]
        tb2 = (boxes_target_tensor[:,:,:,5:10])[second_box_responsible]
        
        regr_in_pred = torch.cat((pb1, pb2), 0)
        regr_in_targ = torch.cat((tb1, tb2), 0)
        
        regression_loss = self.get_regression_loss(regr_in_pred, regr_in_targ)
        
        
        
        
        
        
        
        
        #conf_contains_mask = (contains_object_mask).view(-1)
        
        
        
        """
        pb1 = boxes_pred_tensor[:,:,:,:5][contains_object_mask].reshape((-1, 5))
        pb2 = boxes_pred_tensor[:,:,:,5:10][contains_object_mask].reshape((-1, 5))
        tb1 = boxes_target_tensor[:,:,:,:5][contains_object_mask].reshape((-1, 5))
        tb2 = boxes_target_tensor[:,:,:,5:10][contains_object_mask].reshape((-1, 5))
        
        
        pred = torch.cat((pb1, pb2), 0)
        targ = torch.cat((tb1, tb2), 0)
        """
        
        
        pred = boxes_pred_tensor[contains_object_mask].reshape((-1, 5))
        targ = boxes_target_tensor[contains_object_mask].reshape((-1, 5))
        
        box_target_iou, coo_response_mask = self.find_best_iou_boxes(pred, targ)
        conf_in_pred = pred[coo_response_mask]
        conf_in_target = targ[coo_response_mask]
        conf_iou_in = box_target_iou[coo_response_mask]
        conf_in_target[:,4] = conf_iou_in

        conf_loss = self.get_contain_conf_loss(conf_in_pred, conf_in_target)
        
        
        
        total_loss = class_prediction_loss
        total_loss += no_object_loss
        total_loss += regression_loss
        total_loss += conf_loss
        
        return total_loss



