import cv2
import numpy as np
import PIL.Image
from training import misc

shape = 512
def shiftL(shift,maskIn,imIn,start,end):
    cyclic = 0
    Nshift = shift
    #Ensure minimal width of missing region
    if(start-(256-shift)>0):
        Nshift = start-(256-shift)
    
    ShiftedMask = np.zeros(maskIn.shape, dtype=np.uint8)
    ShiftedIm = np.zeros(imIn.shape, dtype=np.uint8)
    
    #Shift input/mask left
    ShiftedMask[:,0:shape-Nshift] = maskIn[:,Nshift:]
    ShiftedIm[:,0:shape-Nshift] = imIn[:,Nshift:]
    #Complete input/mask on right side
    if(cyclic):
        ShiftedMask[:,shape-shift:] = maskIn[:,:shift]  
        ShiftedIm[:,shape-shift:] = imIn[:,:shift]     
    else:
        ShiftedMask[:,shape-shift:] = cv2.flip(maskIn[:,start-shift:start],1)
        ShiftedIm[:,shape-shift:] = imIn[:,start-shift:start]
    
    
    return ShiftedMask,ShiftedIm
def shiftR(shift,imIn,ori,start,end):
    #Undo ShiftL operation
    if(start-(256-shift)>0):
        shift = start-(256-shift)

    ShiftRight = np.zeros(imIn.shape, dtype=np.uint8)
    ShiftRight[:,shift:shape] = imIn[:,0:shape-shift]
    ShiftRight[:,0:shift] =ori[:,0:shift:]
    return ShiftRight

def findStartMask(mask):

    start = -1
    #Find first row with missing pixels
    for w in range(mask.shape[1]):
        for h in range(mask.shape[0]):
            if(mask[h][w] == 0):
                start = w
                break
        if(start!=-1):
            break
    end = mask.shape[1]
    #Find first row (after start) without missing pixels
    for w in range(start,mask.shape[1]):
        sumcol = 0
        for h in range(mask.shape[0]):
            sumcol+=mask[h][w]
        if(sumcol > 0):
            end = w
            break
   
    return start,end

  
def image_completion_hor_shift(img,mask,Gs,latent):
        shift = 8
        real = PIL.Image.fromarray(img)
        oriSize = real.size
        
        real = real.resize((shape,shape), PIL.Image.ANTIALIAS) 
        real = np.asarray(real)
        
        #Savereal to later reverse shift operation
        saveReal = real.copy()

        maskIn = PIL.Image.fromarray(mask)
        maskIn = maskIn.resize((shape,shape), PIL.Image.ANTIALIAS)
        maskIn = np.asarray(maskIn, dtype=np.float32)

        if(1):
            #Shift for improved result 
            start,end = findStartMask(maskIn[:,:])
            maskInShifted,realShifted = shiftL(shift,maskIn,real,start,end)

            #Preprocess image
            maskInShifted = maskInShifted[np.newaxis]
            realShifted = realShifted.transpose([2, 0, 1])
            realShifted = misc.adjust_dynamic_range(realShifted, [0, 255], [-1, 1])

            #Generate completion
            fake = Gs.run(latent, None, realShifted[np.newaxis], maskInShifted[np.newaxis])[0]
            fake = misc.adjust_dynamic_range(fake, [-1, 1], [0, 255])
            fake = fake.clip(0, 255).astype(np.uint8).transpose([1, 2, 0])
            
            #Reverse shift operation
            fakeShifted = shiftR(shift,fake,saveReal,start,end)
            #Convert back to numpy array
            fakeShifted = PIL.Image.fromarray(fakeShifted)
            fake = PIL.Image.fromarray(fake)
            fakeShifted =fakeShifted.resize(oriSize, PIL.Image.ANTIALIAS)
            fakeShifted = np.asarray(fakeShifted)
            
        return fakeShifted
    
def spatial_outpaint(mask_, video_comp_,Gs):
    mask = mask_
    video_comp = video_comp_       
    
    #Completed size per iteration
    SizeCompletion = 256+64
    
    #Incomming frame + mask
    maskIn_pre = mask.copy()
    arr = video_comp[:, :, :] * 255
    
    #latent vector for co-modulated image completion network
    latent = np.random.randn(1, *Gs.input_shape[1:])
    
    while(1):
        #Find starting point of iteration
        startSegment = maskIn_pre.shape[1]-shape
        for i in range(maskIn_pre.shape[1]):
            #Find first column with masked pixels
            if(sum(maskIn_pre[:,i]) > 0):
                startSegment = i-(shape-SizeCompletion)
                break

        startSegment = min(startSegment,mask.shape[1]-shape)
        startSegmentSquare = max(0,startSegment-(mask.shape[0]-shape))
      
        maskIn = (1-maskIn_pre[:,startSegmentSquare:startSegment+shape].copy()).astype(np.uint8)
        arrIn = ((video_comp[:,startSegmentSquare:startSegment+shape,:].copy())*255).astype(np.uint8)

        #fakeShifted = image_completion_final(arrIn,maskIn,Gs,latent)
        fakeShifted = image_completion_hor_shift(arrIn,maskIn,Gs,latent)
        arr[:,startSegmentSquare:startSegment+shape,:] = fakeShifted
        video_comp[:, :] = arr.copy()/255
        

        #Mark completed region in mask
        maskIn_pre[:,startSegmentSquare:startSegment+shape] = False

        #input()
        
        #Repeat untill all pixels completed
        if(sum(sum(maskIn_pre[:,:])) == 0):
            break
            
    #Set mask of frame as completed
    mask_[:, :] = False
    video_comp_ = video_comp
    return mask_, video_comp_
