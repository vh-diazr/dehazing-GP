
from scipy import misc, math
import cv2
from cv2.ximgproc import guidedFilter
from cv2.ximgproc import guidedFilter
from skimage.restoration import denoise_tv_chambolle
from operator import mul, sub, add
import numpy as np
from skimage.io import imread, imsave

def protectedSqrt(root):
    return math.sqrt(abs(root))
    
def protectedDiv(left, right):
    if (right != 0):
        return left / right
    else:
        return 1

def Max2(a,b):
    return max(a,b)
    
def Min2(a,b):
    return min(a,b)
    
def AbsSum(a,b):
    return abs(a+b)
    
def AbsSub(a,b):
    return abs(a-b)

def pow2(a):
    return pow(a,2)

def protectedLog(a):
    if(a <= 0):
        return 1
    else:
        return math.log(a)
    
def protectedExp(a):
    if(a >= 255):
        a = 1
        return math.exp(a)
    else:
        a = a/255
        return math.exp(a)
        
def myif(a,b,c,d):
    if(a>=b):
        return c
    else:
        return d
    
def estimateAirlight(degraded_I, winSize):
    
    Nr,Nc,Np = degraded_I.shape 
    I1 = np.concatenate((np.flipud(np.fliplr(degraded_I)), np.flipud(degraded_I), np.flipud(np.fliplr(degraded_I))), axis=1)
    I2 = np.concatenate((np.fliplr(degraded_I), degraded_I, np.fliplr(degraded_I)), axis=1)
    I3 = np.concatenate((np.flipud(np.fliplr(degraded_I)), np.flipud(degraded_I), np.flipud(np.fliplr(degraded_I))), axis=1)
    padded_I = np.concatenate( (I1,I2,I3) ,axis=0)
    padded_I = np.double( padded_I[ int(Nr-((winSize-1)/2.0)):int(2.0*Nr+((winSize-1)/2.0)), int(Nc-((winSize-1)/2.0)):int(2.0*Nc+((winSize-1)/2.0)),: ] )
    
    estimate_A = np.zeros([Nr,Nc])
    f_mv = np.zeros([winSize,winSize])
    
    for k in range(Nr):
        for l in range(Nc):
            f_mv = ( padded_I[ k:winSize+k, l:winSize+l, : ] )
            f_max = f_mv.max()
            f_min = f_mv.min()
            u = (f_min + f_max) / 2.0
            v = f_max - f_min
            estimate_A[k,l] = u / (1.0 + v)
        
    
    
    x0,y0 = np.where( estimate_A == estimate_A.max())
    
    A = np.zeros(3)
    A[0] = degraded_I[x0[0], y0[0],0]
    A[1] = degraded_I[x0[0], y0[0],1]
    A[2] = degraded_I[x0[0], y0[0],2]
    estimated_A = 0.333*A[0] + 0.333*A[1] + 0.333*A[2]
    print('Estimate Airlight: Done!')
    return estimated_A

###############################################################################

def dehazeScene(degraded_I, A_est, winSize):
        
    Nr,Nc,Np = degraded_I.shape 
    I1 = np.concatenate((np.flipud(np.fliplr(degraded_I)), np.flipud(degraded_I), np.flipud(np.fliplr(degraded_I))), axis=1)
    I2 = np.concatenate((np.fliplr(degraded_I), degraded_I, np.fliplr(degraded_I)), axis=1)
    I3 = np.concatenate((np.flipud(np.fliplr(degraded_I)), np.flipud(degraded_I), np.flipud(np.fliplr(degraded_I))), axis=1)
    
    padded_I = np.concatenate( (I1,I2,I3) ,axis=0 )
    padded_I = np.double( padded_I[ int(Nr-((winSize-1)/2.0)):int(2.0*Nr+((winSize-1)/2.0)), int(Nc-((winSize-1)/2.0)):int(2.0*Nc+((winSize-1)/2.0)),: ] )
    
    estimated_T = np.zeros([Nr,Nc])
    refined_T = np.zeros([Nr,Nc])
    dehazed_I = np.zeros([Nr,Nc,Np])
    
    for k in range(Nr):
        for l in range(Nc):
            f_v = padded_I[ k:winSize+k, l:winSize+l, : ]
            MAX = np.max(f_v)
            MIN = np.min(f_v)
            RANGE = (MAX - MIN)
            MRANGE = (MAX + MIN)/2.0
            MEAN = np.mean(f_v)
            VAR = np.var(f_v)
            ALPHA = RANGE/(A_est)
            BETA = 1 - ALPHA
            
            #Estimator GP-SD
#            estimated_T[k,l] = protectedExp(add(sub(myif(VAR, AbsSub(RANGE, myif(protectedDiv(protectedExp(add(VAR, 229.5)), protectedDiv(mul(VAR, RANGE), mul(MEAN, MAX))), myif(add(abs(RANGE), protectedSqrt(MEAN)), AbsSum(add(RANGE, sub(MAX, 229.5)), protectedLog(MEAN)), mul(mul(mul(MAX, MIN), MRANGE), Min2(MIN, MRANGE)), add(Min2(ALPHA, VAR), protectedSqrt(VAR))), sub(sub(VAR, abs(ALPHA)), RANGE), add(add(add(AbsSum(MAX, ALPHA), protectedLog(myif(MAX, MAX, sub(MAX, myif(sub(RANGE, RANGE), Max2(BETA, MIN), mul(MIN, MRANGE), Min2(VAR, MRANGE))), add(229.5, abs(AbsSub(mul(MAX, ALPHA), add(MAX, ALPHA))))))), BETA), ALPHA))), MRANGE, myif(sub(RANGE, MAX), Max2(BETA, MIN), mul(MIN, MRANGE), Min2(VAR, MRANGE))), MRANGE), add(sub(RANGE, MAX), protectedLog(myif(protectedDiv(AbsSub(sub(RANGE, MEAN), abs(MIN)), add(sub(RANGE, MAX), protectedLog(myif(protectedDiv(229.5, sub(sub(protectedExp(229.5), Min2(MRANGE, MAX)), Min2(MIN, MRANGE))), myif(protectedDiv(229.5, MAX), protectedLog(MAX), MRANGE, add(MAX, protectedLog(RANGE))), protectedLog(pow2(MAX)), MAX)))), myif(protectedDiv(229.5, 229.5), protectedLog(MAX), sub(sub(VAR, abs(pow2(229.5))), RANGE), add(add(add(sub(RANGE, Max2(MRANGE, MEAN)), protectedLog(myif(MAX, MAX, sub(MAX, sub(RANGE, MAX)), add(229.5, VAR)))), sub(myif(VAR, AbsSub(AbsSub(sub(229.5, MEAN), abs(MIN)), 229.5), MRANGE, myif(sub(RANGE, MAX), Max2(protectedLog(Min2(protectedExp(MRANGE), Min2(VAR, MAX))), MIN), mul(MIN, MRANGE), 229.5)), AbsSub(sub(RANGE, MEAN), abs(MIN)))), ALPHA)), protectedLog(protectedDiv(Max2(MRANGE, RANGE), sub(MRANGE, protectedDiv(AbsSub(MAX, 229.5), pow2(Max2(RANGE, MIN)))))), myif(add(abs(RANGE), add(RANGE, MRANGE)), AbsSum(add(RANGE, ALPHA), protectedLog(MEAN)), mul(mul(MIN, MRANGE), Min2(MIN, MRANGE)), add(Min2(ALPHA, VAR), protectedSqrt(VAR))))))))
            
            #Estimator GP-SN
            estimated_T[k,l] = protectedDiv(protectedExp(myif(229.5, MIN, sub(Min2(MRANGE, VAR), AbsSum(sub(Min2(BETA, add(add(protectedDiv(ALPHA, MRANGE), sub(MIN, MIN)), sub(add(protectedDiv(RANGE, RANGE), AbsSub(RANGE, protectedDiv(pow2(RANGE), protectedExp(AbsSum(Min2(Min2(MEAN, VAR), mul(VAR, VAR)), Min2(pow2(229.5), Min2(229.5, ALPHA))))))), MIN))), MIN), protectedSqrt(sub(Min2(Min2(pow2(229.5), Min2(229.5, ALPHA)), add(add(sub(add(protectedDiv(MEAN, VAR), MEAN), MIN), AbsSub(RANGE, ALPHA)), sub(add(protectedDiv(MIN, Min2(MEAN, VAR)), AbsSub(RANGE, protectedDiv(pow2(RANGE), protectedExp(mul(myif(229.5, 229.5, RANGE, 229.5), protectedDiv(ALPHA, ALPHA)))))), VAR))), MIN)))), add(add(protectedDiv(ALPHA, sub(Min2(Min2(pow2(229.5), Min2(229.5, ALPHA)), add(add(sub(add(protectedDiv(MEAN, VAR), MEAN), MIN), AbsSub(RANGE, ALPHA)), sub(add(protectedDiv(MIN, VAR), AbsSub(RANGE, protectedDiv(pow2(RANGE), protectedExp(mul(myif(229.5, 229.5, RANGE, 229.5), protectedDiv(ALPHA, ALPHA)))))), add(protectedDiv(ALPHA, MRANGE), sub(MIN, MIN))))), MIN)), sub(MIN, MIN)), sub(add(protectedDiv(MIN, ALPHA), AbsSub(RANGE, protectedDiv(pow2(RANGE), protectedExp(MEAN)))), MIN)))), protectedExp(MEAN))
        
    
    print('Estimate Transmission: Done!')
    estimated_T[estimated_T > 1] = 1.0
    estimated_T[estimated_T < 0.01] = 0.1
    
    estimated_T = guidedFilter(np.uint8(degraded_I*255), np.uint8(estimated_T*255), 100, 0.5)/255.0 
    refined_T = denoise_tv_chambolle(estimated_T, weight=0.1, multichannel=False)
    refined_T[refined_T > 1] = 1
    refined_T[refined_T < 0.01] = 0.1
        
    dehazed_I[:,:,0] = ( (degraded_I[:,:,0] - A_est) / refined_T ) + A_est
    dehazed_I[:,:,1] = ( (degraded_I[:,:,1] - A_est) / refined_T ) + A_est 
    dehazed_I[:,:,2] = ( (degraded_I[:,:,2] - A_est) / refined_T ) + A_est
    
    dehazed_I[dehazed_I > 255.0] = 255.0
    dehazed_I[dehazed_I < 0.0] = 0.0
    print('Estimate Dehazed Image: Done!')
    return dehazed_I


if __name__ == "__main__":
    imgName = 'img7'
    degradedScene = np.float64( imread('Images/%s.png' % (imgName) ) )# misc.imread('Images/%s.png' % (imgName)) ) 
    degradedScene = degradedScene[:,:,0:3]
    estimatedA = estimateAirlight(degradedScene, 19)
    dehazed_Img = dehazeScene(degradedScene, estimatedA, 5)
    imsave('Results/%s_outputSN.png' % (imgName), np.uint8(dehazed_Img))



        
        
        
    

