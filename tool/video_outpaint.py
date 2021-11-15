import os
import sys

import time
import argparse
import cv2
import glob
import copy
import numpy as np
import torch
import imageio
import tensorflow as tf
from torchvision import transforms
from torch.utils import data
from torch.autograd import Variable
from training import misc

from PIL import Image, ImageFilter
import scipy.ndimage
from skimage.feature import canny

from dataloaders import PairwiseImg_test as db
from collections import OrderedDict

###
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from RAFT import utils
from RAFT import RAFT
import utils.region_fill as rf
from utils.Poisson_blend import Poisson_blend
from get_flowNN import get_flowNN
from deeplab.siamese_model_conf import CoattentionNet
###
from spatial_inpaint import spatial_inpaint
from spatial_outpaint import spatial_outpaint


TimeCalcFlow = 0
TimeVOS = 0

TimeInpFlowCompletion = 0
TimeOutFlowCompletion = 0

TimeInpTemporal = 0
TimeInpFrame = 0

TimeOutTemporal = 0
TimeOutFrame = 0

perFrames = 0

def create_dir(dir):
    """Creates a directory if not exist.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:#make sure its empty
        images = glob.glob(os.path.join(dir, '*.png')) + \
                    glob.glob(os.path.join(dir, '*.jpg'))
        for f in images:
            os.remove(f)

#COSnet
def convert_state_dict(state_dict):
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove the prefix module.
        state_dict_new[name] = v
        if name == 'linear_e.weight':
            np.save('weight_matrix.npy',v.cpu().numpy())
    return state_dict_new
#COSnet
def getMasks(args,outPath,data_dir,oriShape):
    #Obtain foreground mask using COSNet
    num_classes = 2
    restore_from = '../co_attention.pth'
    seg_save_dir = "./"
    sample_range =1
    masks = []

    first_image = np.array(Image.open(data_dir+'/00000.jpg'))    
    model = CoattentionNet(num_classes=num_classes)
    saved_state_dict = torch.load(restore_from, map_location=lambda storage, loc: storage)
    model.load_state_dict( convert_state_dict(saved_state_dict["model"]) ) 
    model.eval()
    model.cuda()

    db_test = db.PairwiseImg(train=False, inputRes=first_image.shape, db_root_dir=data_dir,  transform=None, sample_range = sample_range) 
    testloader = data.DataLoader(db_test, batch_size= 1, shuffle=False, num_workers=0)

    if not os.path.exists(seg_save_dir):
        os.makedirs(seg_save_dir)

    my_index = -1
    old_temp=''
    for index, batch in enumerate(testloader):
        print('%d processed'%(index))
        target = batch['target']
        temp = batch['seq_name']
        seq_name=temp[0]
        print(seq_name)

        my_index = my_index+1

        output_sum = 0   
        for i in range(0,sample_range):  
            search = batch['search'+'_'+str(i)]
            search_im = search
            output = model(Variable(target, volatile=True).cuda(),Variable(search_im, volatile=True).cuda())

            output_sum = output_sum + output[0].data[0,0].cpu().numpy()
        
        torch.cuda.empty_cache()
        
        output1 = output_sum/sample_range
     
        
        original_shape = first_image.shape 
        output1 = cv2.resize(output1, (original_shape[1],original_shape[0]))
        
                    
        mask = (output1*255).astype(np.uint8)
        #Threshold mask
        for hi in range(mask.shape[0]):
            for wi in range(mask.shape[1]):
                if mask[hi][wi]>1:
                    mask[hi][wi] = 255
                else:
                    mask[hi][wi] = 0
     
        mask = cv2.dilate(mask,np.ones((5,5),np.uint8),iterations = 5)#15
        #mask = cv2.erode(mask,np.ones((5,5),np.uint8),iterations = 10)
        
        mask = cv2.resize(mask, oriShape, interpolation = cv2.INTER_AREA)
        masks.append(mask)
        
    my_index = 0
    for mask in masks:
        mask = Image.fromarray(mask)
        save_dir_res = os.path.join(outPath, 'COSnet')#, args.seq_name)
        old_temp=seq_name
        if not os.path.exists(save_dir_res):
            os.makedirs(save_dir_res)  
        my_index1 = str(my_index).zfill(5)
        seg_filename = os.path.join(save_dir_res, '{}.png'.format(my_index1))
        mask.save(seg_filename)
        my_index+=1
    return masks

def gradient_mask(mask):

    gradient_mask = np.logical_or.reduce((mask,
        np.concatenate((mask[1:, :], np.zeros((1, mask.shape[1]), dtype=np.bool)), axis=0),
        np.concatenate((mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=np.bool)), axis=1)))

    return gradient_mask


def initialize_RAFT(args):
    """Initializes the RAFT model.
    """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to('cuda')
    model.eval()

    return model


def calculate_flow(pathOut, model, video, mode):
    """Calculates optical flow.
    """
    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    nFrame, _, imgH, imgW = video.shape
    Flow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)

    create_dir(os.path.join(pathOut, 'flow', mode + '_flo'))
    create_dir(os.path.join(pathOut, 'flow', mode + '_png'))

    with torch.no_grad():
        for i in range(video.shape[0] - 1):
            print("Calculating {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
            if mode == 'forward':
                # Flow i -> i + 1
                image1 = video[i, None]
                image2 = video[i + 1, None]
            elif mode == 'backward':
                # Flow i + 1 -> i
                image1 = video[i + 1, None]
                image2 = video[i, None]
            else:
                raise NotImplementedError

            _, flow = model(image1, image2, iters=20, test_mode=True)
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            Flow = np.concatenate((Flow, flow[..., None]), axis=-1)

            # Flow visualization.
            flow_img = utils.flow_viz.flow_to_image(flow)
            flow_img = Image.fromarray(flow_img)

            # Saves the flow and flow_img.
            flow_img.save(os.path.join(pathOut, 'flow', mode + '_png', '%05d.png'%i))
            #utils.frame_utils.writeFlow(os.path.join(pathOut, 'flow', mode + '_flo', '%05d.flo'%i), flow)

    return Flow


def extrapolation(args, video_ori, corrFlowF_ori, corrFlowB_ori,first):
    """Prepares the data for video extrapolation.
    """
    imgH, imgW, _, nFrame = video_ori.shape
 
    # Defines new FOV.
    imgH_extr = int(imgH)
    #imgW_extr = int(imgW * (1/(1-1*args.Width)))
    imgW_extr = int(imgW * (1 + args.Width/(1-2*args.Width)))
    if(first):
        W_extrapolated = (imgW_extr-imgW)
        H_extrapolated = (imgH_extr-imgH)
        if((2*W_extrapolated+imgW)%2):
            imgW_extr+=1
        if((2*H_extrapolated+imgH)%2):
            imgH_extr+=1

    H_start = int((imgH_extr - imgH) / 2)
    W_start = 0
    
    # Generates the mask for missing region.
    flow_mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.bool)
    flow_mask[H_start : H_start + imgH, W_start : W_start + int(imgW)] = 0
    # Extrapolates the FOV for video.
    video = np.ones(((imgH_extr, imgW_extr, 3, nFrame)), dtype=np.float32)*255
    video[H_start : H_start + imgH, W_start : W_start + int(imgW), :, :] = video_ori


    # Extrapolates the FOV for flow.
    corrFlowF = np.zeros(((imgH_extr, imgW_extr, 2, nFrame - 1)), dtype=np.float32)
    corrFlowB = np.zeros(((imgH_extr, imgW_extr, 2, nFrame - 1)), dtype=np.float32)
    corrFlowF[H_start : H_start + imgH, W_start : W_start + imgW, :] = corrFlowF_ori
    corrFlowB[H_start : H_start + imgH, W_start : W_start + imgW, :] = corrFlowB_ori

    for i in range(nFrame):
        print("Preparing frame {0}".format(i), '\r', end='')
        video[:, :, :, i] = cv2.inpaint((video[:, :, :, i] * 255).astype(np.uint8), flow_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32)  / 255.

    return video, corrFlowF, corrFlowB, flow_mask, (W_start, H_start), (W_start + imgW, H_start + imgH)


def complete_flow_extrapolation(args, corrFlow, flow_mask, mode,point,edge=None):
    
    imgH, imgW, _, nFrame = corrFlow.shape

    create_dir(os.path.join(args.outroot,'extrapolation', 'flow_comp', mode + '_flo'))
    create_dir(os.path.join(args.outroot,'extrapolation', 'flow_comp', mode + '_png'))
  
    compFlow = np.zeros(((imgH, imgW, 2, nFrame)), dtype=np.float32)
    
    completionSize = int(point)+5
    for i in range(nFrame):
        
        print("Completing {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
        flow = corrFlow[:, :completionSize, :, i]
        flow_mask_img = flow_mask[:, :completionSize, i] if mode == 'forward' else flow_mask[:, :completionSize, i + 1]
        flow_mask_gradient_img = gradient_mask(flow_mask_img)
        
        #HIER (0):#
        if edge is not None:
            
            gradient_x = np.concatenate((np.diff(flow, axis=1), np.zeros((imgH, 1, 2), dtype=np.float32)), axis=1)
            gradient_y = np.concatenate((np.diff(flow, axis=0), np.zeros((1, completionSize, 2), dtype=np.float32)), axis=0)

            # concatenate gradient_x and gradient_y
            gradient = np.concatenate((gradient_x, gradient_y), axis=2)

            # We can trust the gradient outside of flow_mask_gradient_img
            # We assume the gradient within flow_mask_gradient_img is 0.
            gradient[flow_mask_gradient_img, :] = 0

            # Complete the flow
            imgSrc_gy = gradient[:, :completionSize, 2 : 4]
            imgSrc_gy = imgSrc_gy[0 : imgH - 1, :completionSize, :]
            imgSrc_gx = gradient[:, :completionSize, 0 : 2]
            imgSrc_gx = imgSrc_gx[:, 0 : completionSize - 1, :] 

            compFlow[:, :completionSize, :, i] = Poisson_blend(flow, imgSrc_gx, imgSrc_gy, flow_mask_img, edge[:, :completionSize, i])
            compFlow[:, :point, :, i] = flow[:,:point,:]
        else:
    
            flow[:, :, 0] = rf.regionfill(flow[:, :, 0], flow_mask_img)
            flow[:, :, 1] = rf.regionfill(flow[:, :, 1], flow_mask_img)
            compFlow[:, :completionSize, :, i] = flow
        
        
        for ii in range(completionSize-5,compFlow.shape[1]):
            compFlow[:,ii,:,i] = compFlow[:,completionSize-1,:,i].copy()
            
        
        # Flow visualization.
        flow_img = utils.flow_viz.flow_to_image(compFlow[:, :, :, i])
        flow_img = Image.fromarray(flow_img)

        # Saves the flow and flow_img.
        flow_img.save(os.path.join(args.outroot,'extrapolation', 'flow_comp', mode + '_png', '%05d.png'%i))
        #utils.frame_utils.writeFlow(os.path.join(args.outroot,'extrapolation', 'flow_comp', mode + '_flo', '%05d.flo'%i), compFlow[:, :, :, i])

    return compFlow
def complete_flow(args, corrFlow, flow_mask, mode):
    imgH, imgW, _, nFrame = corrFlow.shape

    create_dir(os.path.join(args.outroot,'object_removal', 'flow_comp', mode + '_flo'))
    create_dir(os.path.join(args.outroot,'object_removal', 'flow_comp', mode + '_png'))

    compFlow = np.zeros(((imgH, imgW, 2, nFrame)), dtype=np.float32)

    for i in range(nFrame):
        print("Completing {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
        flow = corrFlow[:, :, :, i]
        flow_mask_img = flow_mask[:, :, i] if mode == 'forward' else flow_mask[:, :, i + 1]
        flow_mask_gradient_img = gradient_mask(flow_mask_img)

        flow[:, :, 0] = rf.regionfill(flow[:, :, 0], flow_mask_img)
        flow[:, :, 1] = rf.regionfill(flow[:, :, 1], flow_mask_img)
        compFlow[:, :, :, i] = flow

        # Flow visualization.
        flow_img = utils.flow_viz.flow_to_image(compFlow[:, :, :, i])
        flow_img = Image.fromarray(flow_img)

        # Saves the flow and flow_img.
        flow_img.save(os.path.join(args.outroot,'object_removal', 'flow_comp', mode + '_png', '%05d.png'%i))
        #utils.frame_utils.writeFlow(os.path.join(args.outroot,'object_removal', 'flow_comp', mode + '_flo', '%05d.flo'%i), compFlow[:, :, :, i])

    return compFlow


def edge_completion(args, corrFlow, flow_mask, mode,start_point,corrFlowFori,maskVOS):
    """Calculate flow edge and complete it.
    """

    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    imgH, imgW, _, nFrame = corrFlow.shape
    Edge = np.empty(((imgH, imgW, 0)), dtype=np.float32)

    for i in range(nFrame):
        print("Completing {0} flow edge {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
        flow_mask_img = flow_mask[:, :, i] if mode == 'forward' else flow_mask[:, :, i + 1]

        flow_img_gray = (corrFlowFori[:, :, 0, i] ** 2 + corrFlowFori[:, :, 1, i] ** 2) ** 0.5
        flow_img_gray = flow_img_gray / flow_img_gray.max()

        edge_corr = np.zeros((flow_img_gray.shape[0],flow_img_gray.shape[1]), dtype=np.uint8)#canny(flow_img_gray, sigma=2)#, mask=(1 - flow_mask_img).astype(np.bool))
        new = np.zeros((corrFlow.shape[0],corrFlow.shape[1]), dtype=np.uint8)
        new[:,:edge_corr.shape[1]] = edge_corr
        edge_corr = new
        W = start_point[0]

        if(0):
            startEdge,endEdge = imgH,0
            
            for ii in range(imgH):
                if(edge_corr[ii,W-2]):
                    startEdge = max(0,ii-10)
                    break
                
            for ii in range(imgH-1,-1,-1):
                if(edge_corr[ii,W-2]):                
                    endEdge = min(imgH,ii+10)
                    break
            for ii in range(startEdge,endEdge):
                
                edge_corr[ii,W-2] = True
                edge_corr[ii,W-3] = True
                edge_corr[ii,W-4] = True
                edge_corr[ii,W-5] = True
        else:
            mask_pred = cv2.dilate(maskVOS[:,:,i].astype('uint8'),np.ones((5,5),np.uint8),iterations = 5)
            for ii in range(maskVOS.shape[0]):
                for jj in range(maskVOS.shape[1]):
                    if(maskVOS[ii,jj,i]>0):
                        edge_corr[ii,jj] = True
                    
        edge_completed = edge_corr
        cv2.imwrite("./edges/edge"+str(i)+".jpg", edge_corr*255)
        
        
        Edge = np.concatenate((Edge, edge_completed[..., None]), axis=-1)

    return Edge


def getFlowVideo(args,pathIn):
    pathOut = os.path.join(args.outroot, 'object_removal')
    # Find frames.
    filename_list = glob.glob(os.path.join(pathIn, '*.png')) + \
                    glob.glob(os.path.join(pathIn, '*.jpg'))
    
    filename_list = filename_list[:75]
    
    # Obtains imgH, imgW and nFrame.
    if(len(filename_list)==0):
        print("No images in directory:",pathIn)
        return

    nFrame = len(filename_list)

    # Load video.
    video,video2 = [],[]
    videoRaft,videoRaft2 = [],[]
    H_scale,Width = None,None
    scaleRaft = 1
    
    i=0
    for filename in sorted(filename_list):
        i+=1
        img = np.array(Image.open(filename)).astype(np.uint8)
        print(img.shape)
        
        

        if(img.shape[0]>512):
            img = cv2.resize(img, (int(img.shape[1]* 512/img.shape[0]),512), interpolation = cv2.INTER_AREA)
        W = img.shape[1] 
        if(args.replace):
            print(W)
            img = img[:,int(args.Width*W):int(W-args.Width*W),:]
       
        #Input of RAFT (optical flow estimation) must be devisable by 8
        #Reshape dimension to closest multiple of 8

        W, H,_ = img.shape

        scaleRaft = max((W*H)/1000000,1) # Scale to Limit size of image on gpu to prevent out of memory

        print(scaleRaft,H/scaleRaft,W/scaleRaft)
        print("img:",filename,img.shape)
        for i in range(2560):
            lower,upper = i*2**3,(i+1)*2**3
            if(lower<=H/scaleRaft<=upper):
                if(H-lower<upper-H):
                    H_scale = lower
                else:
                    H_scale = upper
            if(lower<=W/scaleRaft<=upper):
                if(W-lower<upper-W):
                    Width = lower  
                else:
                    Width = upper
        imgRaft = cv2.resize(img, (H_scale,Width), interpolation = cv2.INTER_AREA) 
 
        print(H_scale,Width,imgRaft.shape)
        video.append(torch.from_numpy(img).permute(2, 0, 1).float())
        videoRaft.append(torch.from_numpy(imgRaft).permute(2, 0, 1).float())
        video2.append(torch.from_numpy(cv2.flip(img,1)).permute(2, 0, 1).float())
        videoRaft2.append(torch.from_numpy(cv2.flip(imgRaft,1)).permute(2, 0, 1).float())
    # Video on gpu
    video = torch.stack(video, dim=0)
    videoRaft = torch.stack(videoRaft, dim=0)
    videoRaft = videoRaft.to('cuda')
    
    # Flow model.
    RAFT_model = initialize_RAFT(args)
    # Calcutes the corrupted flow.
    print('\nStart flow prediction.')
    print(videoRaft.shape)
    
    t1 = time.perf_counter()   
    corrFlowF = calculate_flow(pathOut, RAFT_model, videoRaft, 'forward')
    corrFlowB = calculate_flow(pathOut, RAFT_model, videoRaft, 'backward')
    t2 = time.perf_counter()   
    print("time spend calculate flow:",t2-t1)
    global TimeCalcFlow
    TimeCalcFlow += t2-t1
        
    corrFlowF[:,:,:,:]*=scaleRaft
    corrFlowB[:,:,:,:]*=scaleRaft
    print('\nFinish flow prediction.')

    # Convert to b,g,r,frame.
    video = video.permute(2, 3, 1, 0).numpy()[:, :, ::-1, :] / 255.
    videoRaft = videoRaft.permute(2, 3, 1, 0).cpu().numpy()[:, :, ::-1, :] / 255.
 
    corrFlowF_Resized = []
    corrFlowB_Resized = []
    
    for i in range(corrFlowF.shape[3]):
        corrFlowF_Resized.append(cv2.resize(corrFlowF[:,:,:,i], (H,W), interpolation = cv2.INTER_AREA))
        corrFlowB_Resized.append(cv2.resize(corrFlowB[:,:,:,i], (H,W), interpolation = cv2.INTER_AREA))
        
    corrFlowF = np.array(corrFlowF_Resized)
    corrFlowB = np.array(corrFlowB_Resized)
    
    corrFlowF =  np.transpose(corrFlowF, axes=[1, 2, 3, 0])
    corrFlowB =  np.transpose(corrFlowB, axes=[1, 2, 3, 0])
 
    return video,corrFlowF,corrFlowB

def setup_video_completion(args,video,corrFlowF,corrFlowB,outPath,FG_removal,first):   
    print(video.shape)

    imgH, imgW,channels,nFrame = video.shape
    global perFrames
    perFrames = nFrame
    if(FG_removal):
        mask = []
        with torch.no_grad():
            t1 = time.perf_counter()   
            mask = getMasks(args,outPath,os.path.join(args.outroot, 'input_scaled'),(imgW,imgH))#COSnet
            t2 = time.perf_counter()   
            print("time spend VOS:",t2-t1)
            global TimeVOS
            TimeVOS += t2-t1
        torch.cuda.empty_cache()
           
        flow_mask =  np.copy(mask)
        # mask indicating the missing region in the video.
        mask = np.stack(mask, -1).astype(np.bool)
        flow_mask = np.stack(flow_mask, -1).astype(np.bool)
        end_point = video.shape[1]
        
    else:#Outpaint
        #mask,flow_mask,end_point = video_extrapolation(args,video,corrFlowF,corrFlowB)
        video, corrFlowF, corrFlowB, flow_mask, start_point, end_point = extrapolation(args, video, corrFlowF, corrFlowB,first)
        imgH, imgW = video.shape[:2]

        # mask indicating the missing region in the video.
        mask = np.tile(flow_mask[..., None], (1, 1, nFrame))
        flow_mask = np.tile(flow_mask[..., None], (1, 1, nFrame))

    return video,mask,corrFlowF,corrFlowB,end_point
    
def video_completion(Gs,args,video,mask,videoFlowF,videoFlowB,end_point,maskVOS,FG_removal):
    torch.cuda.empty_cache()
    imgH, imgW,channels,nFrame = video.shape
    video_comp = video
    iter = 0
    mask_tofill = mask
    for i in range(nFrame):
        # Dilate 15 pixel so that all known pixel is trustworthy
        mask_tofill[:,:,i] = scipy.ndimage.binary_dilation(mask_tofill[:,:,i], iterations=8)
    

    flow_mask =  np.copy(mask)
    
    FlowF_edge, FlowB_edge = None, None
    if(not(FG_removal)):
        # Edge completion.
        FlowF_edge = edge_completion(args, videoFlowF, flow_mask, 'forward',end_point,videoFlowF,maskVOS)
        FlowB_edge = edge_completion(args, videoFlowB, flow_mask, 'backward',end_point,videoFlowB,maskVOS)
        print('\nFinish edge completion.')
    # Completes the flow.
    t1 = time.perf_counter() 
    if(not(FG_removal)):# args.mode == 'video_extrapolation':
        videoFlowF = complete_flow_extrapolation(args, videoFlowF, flow_mask, 'forward', maskVOS.shape[1],FlowF_edge)
        videoFlowB = complete_flow_extrapolation(args, videoFlowB, flow_mask, 'backward',maskVOS.shape[1], FlowB_edge)
    else:
        videoFlowF = complete_flow(args, videoFlowF, flow_mask, 'forward')
        videoFlowB = complete_flow(args, videoFlowB, flow_mask, 'backward') 
        
    t2 = time.perf_counter()   
    print("time spend flow completion outpaint:",t2-t1)
    global TimeOutFlowCompletion
    TimeOutFlowCompletion += t2-t1

    print('\nFinish flow completion.')
    
    timeTemporalProp = 0
    timeFrameComp = 0
    
    if(FG_removal):
        create_dir(os.path.join(args.outroot,'object_removal', 'frame_comp_' + str(iter)))
    else:
        create_dir(os.path.join(args.outroot,'extrapolation', 'frame_comp_' + str(iter)))
    
    for i in range(nFrame):
        #mask_tofill[:, :, i] = scipy.ndimage.binary_dilation(mask_tofill[:, :, i], iterations=2)
        img = video_comp[:, :, :, i] * 255
        # Green indicates the regions that are not filled yet.
        img[mask_tofill[:, :, i]] = [0, 255, 0]
        #print(os.path.join(args.outroot, 'frame_comp_' + str(iter), '%05d.jpg'%i))
        if(FG_removal):
            cv2.imwrite(os.path.join(args.outroot,'object_removal', 'frame_comp_' + str(iter), '%05d.jpg'%i), img)    
        else:
            cv2.imwrite(os.path.join(args.outroot,'extrapolation', 'frame_comp_' + str(iter), '%05d.jpg'%i), img)  
    iter += 1        

    
    # Initial Color propagation.
    print('\nInitial Color propagation.')
    t1 = time.perf_counter()
    video_comp[:,:,:,:], mask_tofill[:,:,:], _ ,sumNewPixels = get_flowNN(args,
                                      video_comp[:,:,:,:],
                                      mask_tofill[:,:,:],
                                      videoFlowF[:,:,:,:],
                                      videoFlowB[:,:,:,:]
                                      )
    t2 = time.perf_counter()   
    timeTemporalProp+= (t2-t1)
    
  

    """
    #Save images of completion steps
    for i in range(nFrame):
            mask_tofill[:, :, i] = scipy.ndimage.binary_dilation(mask_tofill[:, :, i], iterations=2)
            img = video_comp[:, :, :, i] * 255
            # Green indicates the regions that are not filled yet.
            img[mask_tofill[:, :, i]] = [0, 255, 0]
            cv2.imwrite(os.path.join(args.outroot,'extrapolation', 'frame_comp_' + str(iter), '%05d.png'%i), img)
    """
    prevKeyFrameInd = -1
    # We iteratively complete the video.
    while(np.sum(mask_tofill) > 0):
        #Frame with most masked pixels first 
        t1 = time.perf_counter()  
        maxS = -1
        for ii in range(mask.shape[2]):
            sizeToFill = np.sum(mask[:,:,ii])
            print(sizeToFill)
            if(sizeToFill>maxS and sizeToFill !=0):
                keyFrameInd = ii
                maxS = sizeToFill

        prevKeyFrameInd = keyFrameInd

        if(FG_removal):
            mask_tofill[:, :, keyFrameInd], video_comp[:, :, :, keyFrameInd] = spatial_inpaint(mask_tofill[:, :, keyFrameInd], video_comp[:, :, :, keyFrameInd],Gs)
        else:
            mask_tofill[:, :, keyFrameInd], video_comp[:, :, :, keyFrameInd] = spatial_outpaint(mask_tofill[:, :, keyFrameInd], video_comp[:, :, :, keyFrameInd],Gs)
        
        t2 = time.perf_counter()   
        timeFrameComp+= (t2-t1)

        if(maxS>0.0004*video_comp.shape[0]*video_comp.shape[1]):#Only temprolly propagate significant completions to save time
            t1 = time.perf_counter()   
            # Color propagation of frame completion.
            video_comp[:,:,:,:], mask_tofill[:,:,:], _ ,sumNewPixels= get_flowNN(args,
                                              video_comp[:,:,:,:],
                                              mask_tofill[:,:,:],
                                              videoFlowF[:,:,:,:],
                                              videoFlowB[:,:,:,:]
                                             )
            t2 = time.perf_counter()   
            timeTemporalProp+= (t2-t1)
        #Save results of each iteration
        """
        if(FG_removal):
            create_dir(os.path.join(args.outroot,'object_removal', 'frame_comp_' + str(iter)))
        else:
            create_dir(os.path.join(args.outroot,'extrapolation', 'frame_comp_' + str(iter)))
        for i in range(nFrame):
            #Save images of completion steps
            #mask_tofill[:, :, i] = scipy.ndimage.binary_dilation(mask_tofill[:, :, i], iterations=2)
            img = video_comp[:, :, :, i] * 255
            # Green indicates the regions that are not filled yet.
            img[mask_tofill[:, :, i]] = [0, 255, 0]
            #print(os.path.join(args.outroot, 'frame_comp_' + str(iter), '%05d.jpg'%i))
            if(FG_removal):
                cv2.imwrite(os.path.join(args.outroot,'object_removal', 'frame_comp_' + str(iter), '%05d.jpg'%i), img)    
            else:
                cv2.imwrite(os.path.join(args.outroot,'extrapolation', 'frame_comp_' + str(iter), '%05d.jpg'%i), img)    """
        iter += 1 
        
    print("time spent temporalProp:",timeTemporalProp)
    print("time spent frame comp:",timeFrameComp)
    global TimeOutTemporal
    TimeOutTemporal += timeTemporalProp
    global TimeOutFrame
    TimeOutFrame = timeFrameComp
  
    return video_comp
def saveResult(video_path,video):
    imgH, imgW,channels,nFrame = video.shape
    
    create_dir(video_path)
    video_ = (video * 255).astype(np.uint8).transpose(3, 0, 1, 2)[:, :, :, ::-1]
    for i in range(nFrame):
        img = video[:, :, :, i] * 255
        cv2.imwrite(os.path.join(video_path, '%05d.jpg'%i), img)

def saveResultVid(video_path,video):
    imgH, imgW,channels,nFrame = video.shape
    
    create_dir(video_path)
    video_ = (video).astype(np.uint8).transpose(3, 0, 1, 2)[:, :, :, ::-1]
    for i in range(nFrame):
        img = video[:, :, :, i]
        cv2.imwrite(os.path.join(video_path, '%05d.jpg'%i), img)
    imageio.mimwrite(os.path.join(video_path, 'final.mp4'), video_, fps=12, quality=8, macro_block_size=1)


def main(args):

    in_path = args.path
    out_path = os.path.join(args.outroot, 'object_removal')

    video,corrFlowF,corrFlowB = getFlowVideo(args,in_path) 
    saveResult(os.path.join(args.outroot, 'input_video'),video)
    
    H0, W0, c, nframe = video.shape
        
    scaleVOS = 1#(W0*H0)/200000 # Scale to Limit size of image on gpu to prevent out of memory
    videoScaled = []
    for i in range(nframe):
        videoScaled.append(cv2.resize(video[:,:,:,i], (int(W0/scaleVOS),int(H0/scaleVOS)), interpolation = cv2.INTER_AREA))
    videoScaled = np.array(videoScaled)
    videoScaled = np.transpose(videoScaled, (1,2,3,0))
    saveResult(os.path.join(args.outroot, 'input_scaled'),videoScaled)
    del videoScaled
    
    video,maskVOS,videoFlowF,videoFlowB,end_point = setup_video_completion(args,video,corrFlowF,corrFlowB,out_path,1,0)  
        
    # Load Image inpainting model.
    _, _, Gs = misc.load_pkl('../co-mod-gan-places2-050000.pkl')
    
    video_comp = video_completion(Gs,args,video,maskVOS,videoFlowF,videoFlowB,end_point,None,1)
    video_path = os.path.join(out_path, 'final')
    saveResult(video_path,video_comp)
    
    out_path = os.path.join(args.outroot, 'extrapolation') 
    video,corrFlowF,corrFlowB = video_comp.copy(),videoFlowF.copy(),videoFlowB.copy()
    video2,corrFlowF2,corrFlowB2 = video_comp.copy(),videoFlowF.copy(),videoFlowB.copy()
    del video_comp
    
    #1: original video, mask, flow
    #2: 1 mirrored vertically
    maskVOS2 = maskVOS.copy()*1
    for i in range(nframe):
        video2[:,:,:,i] = cv2.flip(video2[:,:,:,i],1)
        maskVOS[:,:,i] = maskVOS[:,:,i]*1 #(maskVOS[:,:,i]*1).copy()
        maskVOS2[:,:,i] = cv2.flip((maskVOS2[:,:,i]*1).copy(),1)
        if(i<nframe-1):
            corrFlowF2[:,:,:,i] = cv2.flip(videoFlowF[:,:,:,i],1)
            corrFlowB2[:,:,:,i] = cv2.flip(videoFlowB[:,:,:,i],1)
            corrFlowF2[:,:,0,i]*=-1
            corrFlowB2[:,:,0,i]*=-1
        
    del videoFlowF,videoFlowB        
    #inputvideo, mask,flow for R(ight) side completion
    video1,mask1,videoFlowF1,videoFlowB1,end_point = setup_video_completion(args,video,corrFlowF,corrFlowB,out_path,0,1)     
    video_comp1 = video_completion(Gs,args,video1,mask1,videoFlowF1,videoFlowB1,end_point,maskVOS,0)
    
    del video1,mask1,videoFlowF1,videoFlowB1,maskVOS
    #inputvideo, mask,flow for L(eft) side completion
    video2,mask2,videoFlowF2,videoFlowB2,end_point2 = setup_video_completion(args,video2,corrFlowF2,corrFlowB2,out_path,0,0)
    video_comp2 = video_completion(Gs,args,video2,mask2,videoFlowF2,videoFlowB2,end_point2,maskVOS2,0)
        
    del video2,mask2,videoFlowF2,videoFlowB2,maskVOS2
    
    #Save result of each completion
    video_path = os.path.join(args.outroot, 'extrapolation', 'final1')
    saveResult(video_path,video_comp1)  
    video_path = os.path.join(args.outroot, 'extrapolation', 'final2')
    saveResult(video_path,video_comp2)
    
    ##################################################################
    ###Combine both completions with the input video + post processing   
    
    size_comp = video_comp2.shape[1]-W0  
    total_video = np.zeros((H0,video_comp1.shape[1] + video_comp2.shape[1] - W0,c,nframe), dtype=np.float)
    total_video_blur = np.zeros((H0,video_comp1.shape[1] + video_comp2.shape[1] - W0,c,nframe), dtype=np.float)
    
    #Set left and right side of result    
    typeBlur = 0
    for i in range(nframe):
        lefthalf = cv2.flip(video_comp2[:,W0:,:,i]*255,1) 
        righthalf = video_comp1[:,W0:,:,i]*255
        total_video[:,:size_comp,:,i] = lefthalf
        total_video[:,size_comp+W0:,:,i] = righthalf
        if(typeBlur == 1):
            sizeK = 5
            kernel = np.ones((sizeK,sizeK),np.float32)/(sizeK*sizeK)
            lefthalf = cv2.filter2D(lefthalf,-1,kernel)
            righthalf = cv2.filter2D(righthalf,-1,kernel)
        elif(typeBlur == 1):
            lefthalf = (Image.fromarray(lefthalf.astype(np.uint8))).filter(ImageFilter.BLUR)
            righthalf = (Image.fromarray(righthalf.astype(np.uint8))).filter(ImageFilter.BLUR)
        else:
            lefthalf = (Image.fromarray(lefthalf.astype(np.uint8))).filter(ImageFilter.GaussianBlur(2))
            righthalf = (Image.fromarray(righthalf.astype(np.uint8))).filter(ImageFilter.GaussianBlur(2))

             
        total_video_blur[:,:size_comp,:,i] = lefthalf
        total_video_blur[:,size_comp+W0:,:,i] = righthalf
        
        
    cv2.imwrite(os.path.join(out_path, 'final_total', 'test0.png'), total_video[:,:,:,i]) 
    filename_list2 = glob.glob(os.path.join(args.outroot, 'input_video', '*.png')) + \
                    glob.glob(os.path.join(args.outroot, 'input_video', '*.jpg'))
    #Set center of result to original video
    i = 0
    for filename in sorted(filename_list2):
        b, g, r = Image.open(filename).split()
        im = Image.merge("RGB", (r, g, b))
        img = np.array(im)
        total_video[:,size_comp:size_comp+W0,:,i] = img
        total_video_blur[:,size_comp:size_comp+W0,:,i] = img
        i+=1
        
    #Save results
    video_path = os.path.join(args.outroot,'final_total')
    saveResultVid(video_path,total_video)
    video_path = os.path.join(args.outroot,'final_blurred')
    saveResultVid(video_path,total_video_blur)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    parser = argparse.ArgumentParser()
    # video completion
    #parser.add_argument('--edge_guide', action='store_true', help='Whether use edge as guidance to complete flow')
    
    parser.add_argument('--mode', default='object_removal', help="modes: object_removal / video_extrapolation")
    parser.add_argument('--path', default='../frames/', help="dataset for evaluation")
    parser.add_argument('--outroot', default='../results/frames', help="output directory")
    parser.add_argument('--consistencyThres', dest='consistencyThres', default=np.inf, type=float, help='flow consistency error threshold')
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
    
    # RAFT
    parser.add_argument('--model', default='../raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    
    # extrapolation
    parser.add_argument('--Width', dest='Width', default=2, type=float, help='Amount of pixels extrapolated to each side')
    parser.add_argument('--replace', action='store_true', help="Remove 'Width' pixels and recomplete them (for evaluation)")
    args = parser.parse_args()

    with torch.no_grad():
        t1 = time.perf_counter()   
        main(args)
        t2 = time.perf_counter()   
    
print(TimeCalcFlow)
print(TimeVOS)
print(TimeInpFlowCompletion)
print(TimeOutFlowCompletion)
print(TimeOutTemporal)
print(TimeOutFrame)
print(t2-t1)
"""
#Log Times to calculate average
with open("times.txt", "a") as file_object:
    file_object.write(str(args.path)+"\n")
    file_object.write(str(perFrames)+"\n")
    file_object.write(str(TimeCalcFlow)         +"     "+str(TimeCalcFlow/perFrames)+"\n")
    file_object.write(str(TimeVOS)              +"     "+str(TimeVOS/perFrames)+"\n")
    file_object.write(str(TimeInpFlowCompletion)+"     "+str(TimeInpFlowCompletion/perFrames)+"\n")
    file_object.write(str(TimeOutFlowCompletion)+"     "+str(TimeOutFlowCompletion/perFrames)+"\n")
    file_object.write(str(TimeOutTemporal)      +"     "+str(TimeOutTemporal/perFrames)+"\n")
    file_object.write(str(TimeOutFrame)         +"     "+str(TimeOutFrame/perFrames)+"\n")
    file_object.write(str(t2-t1)+"\n\n")
"""


