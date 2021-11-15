import os
import cv2
import numpy as np
import PIL.Image
from training import misc

shape = 512 # co-modulated trained on image of 512 pixels

def findStartMask(mask):
    start = -1
    for w in range(mask.shape[1]):
        for h in range(mask.shape[0]):
            if(mask[h][w] == 0):
                start = w
                break
        if(start!=-1):
            break
    return start

def findStartHeight(mask):
    shape = mask.shape[1]
    start = -1
    for h in range(mask.shape[0]):
        for w in range(mask.shape[1]):
            if(mask[h][w] == 0):
                start = h
                break
        if(start!=-1):
            break
    return start
    
def spatial_inpaint(mask, video_comp,Gs):
    height,width = mask.shape
    size = min(height,width)
    wStart = max(findStartMask(mask) - 32,0)
    hStart = max(findStartHeight(mask) - 32,0)
    hEnd = hStart + size 
    wEnd = wStart + size
    
    latent = np.random.randn(1, *Gs.input_shape[1:])
    while(1):#top to bottem
        while(1):#left to right
            arrIn = (video_comp[hStart:hEnd,wStart:wEnd,:].copy())*255
            arrIn = arrIn.astype(np.uint8)
            real = PIL.Image.fromarray(arrIn)
            oriSize = real.size
            real = real.resize((512,512), PIL.Image.ANTIALIAS)
            real = real.convert('RGB')
            real = np.asarray(real)
            
            maskIn = 1-mask[hStart:hEnd,wStart:wEnd].copy()
            maskIn = maskIn.astype(np.uint8)
            maskIn = PIL.Image.fromarray(maskIn)
            maskIn = maskIn.resize((512,512), PIL.Image.ANTIALIAS)
            maskIn = np.asarray(maskIn, dtype=np.float32)
           
            maskIn = maskIn[np.newaxis]
            real = real.transpose([2, 0, 1])
            real = misc.adjust_dynamic_range(real, [0, 255], [-1, 1])
            
            fake = Gs.run(latent, None, real[np.newaxis], maskIn[np.newaxis])[0]
            fake = misc.adjust_dynamic_range(fake, [-1, 1], [0, 255])
            fake = fake.clip(0, 255).astype(np.uint8).transpose([1, 2, 0])
            fake = PIL.Image.fromarray(fake)

            fake =fake.resize(oriSize, PIL.Image.ANTIALIAS)
            fake = np.asarray(fake)/255
            video_comp[hStart:hEnd,wStart:wEnd,:] = fake
            mask[hStart:hEnd, wStart:wEnd] = False
            if(wEnd == width):
                break
            wStart = wEnd
            wEnd = wStart + size 
            if(wEnd >width):
                wEnd = width
                wStart = wEnd - size   
        if(hEnd == height):
            break
        
        hStart = hEnd
        hEnd = hStart + size 
        if(hEnd >height):
            hEnd = height
            hStart = hEnd - size
         
    return mask, video_comp
    
