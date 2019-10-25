
from scipy import misc, math
from cv2.ximgproc import guidedFilter
from skimage.restoration import denoise_tv_chambolle
from operator import mul, sub, add
import numpy as np

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

def dehazeScene(degraded_I, A_est):
        
    Nr,Nc,Np = degraded_I.shape 
    winSize = 3 #The evolved estiators GP-PD and GP-PN were designed for a fixed window size = 3x3 pixels
    I1 = np.concatenate((np.flipud(np.fliplr(degraded_I)), np.flipud(degraded_I), np.flipud(np.fliplr(degraded_I))), axis=1)
    I2 = np.concatenate((np.fliplr(degraded_I), degraded_I, np.fliplr(degraded_I)), axis=1)
    I3 = np.concatenate((np.flipud(np.fliplr(degraded_I)), np.flipud(degraded_I), np.flipud(np.fliplr(degraded_I))), axis=1)
    
    padded_I = np.concatenate( (I1,I2,I3) ,axis=0 )
    padded_I = np.double( padded_I[ int(Nr-((winSize-1)/2.0)):int(2.0*Nr+((winSize-1)/2.0)), int(Nc-((winSize-1)/2.0)):int(2.0*Nc+((winSize-1)/2.0)),: ] )
    
    estimated_T = np.zeros([Nr,Nc])
    refined_T = np.zeros([Nr,Nc])
    dehazed_I = np.zeros([Nr,Nc,Np])
    N = Np*(winSize**2)
    for k in range(Nr):
        for l in range(Nc):
            f_v = padded_I[ k:winSize+k, l:winSize+l, : ]
            I_mv = np.reshape(f_v, N)
            ARG = list(I_mv)
            ARG0 = ARG[0]
            ARG1 = ARG[1]
            ARG2 = ARG[2]
            ARG3 = ARG[3]
            ARG4 = ARG[4]
            ARG5 = ARG[5]
            ARG6 = ARG[6]
            ARG7 = ARG[7]
            ARG8 = ARG[8]
            ARG9 = ARG[9]
            ARG10 = ARG[10]
            ARG11 = ARG[11]
            ARG12 = ARG[12]
            ARG13 = ARG[13]
            ARG14 = ARG[14]
            ARG15 = ARG[15]
            ARG16 = ARG[16]
            ARG17 = ARG[17]
            ARG18 = ARG[18]
            ARG19 = ARG[19]
            ARG20 = ARG[20]
            ARG21 = ARG[21]
            ARG22 = ARG[22]
            ARG23 = ARG[23]
            ARG24 = ARG[24]
            ARG25 = ARG[25]
            ARG26 = ARG[26]
            
            #Estimator GP-PD
            estimated_T[k,l] = protectedExp(myif(ARG5, sub(ARG16, ARG14), myif(ARG5, sub(ARG4, ARG2), sub(protectedLog(protectedSqrt(protectedExp(sub(sub(ARG25, myif(myif(ARG7, protectedDiv(ARG2, ARG25), protectedDiv(abs(Min2(protectedLog(ARG12), protectedExp(ARG7))), abs(abs(229.5))), protectedDiv(protectedExp(protectedSqrt(ARG17)), 229.5)), protectedLog(protectedExp(protectedSqrt(sub(ARG2, ARG1)))), mul(myif(protectedDiv(ARG25, ARG17), ARG7, ARG8, Max2(myif(ARG10, ARG17, Max2(add(ARG1, ARG25), myif(ARG9, ARG23, ARG23, ARG6)), ARG11), protectedDiv(ARG25, ARG20))), ARG17), protectedDiv(mul(Min2(ARG4, sub(protectedExp(ARG16), abs(abs(sub(ARG13, 229.5))))), ARG21), protectedExp(ARG4)))), mul(ARG1, protectedSqrt(mul(ARG4, Min2(ARG4, ARG17)))))))), Min2(abs(abs(add(sub(ARG16, ARG8), Min2(ARG25, Min2(Min2(ARG4, ARG17), abs(abs(add(sub(ARG4, ARG2), Min2(ARG25, ARG5))))))))), ARG14)), Min2(protectedExp(ARG11), myif(ARG9, ARG0, ARG3, Min2(ARG25, add(myif(myif(protectedDiv(ARG24, Max2(ARG19, ARG25)), ARG25, ARG14, ARG5), sub(ARG19, abs(ARG19)), protectedSqrt(protectedDiv(protectedDiv(ARG4, ARG2), protectedDiv(ARG17, ARG20))), abs(229.5)), ARG7))))), Min2(protectedExp(ARG11), protectedLog(sub(ARG23, sub(ARG6, protectedExp(Max2(myif(protectedDiv(ARG25, ARG17), mul(protectedDiv(ARG24, protectedDiv(ARG6, ARG0)), mul(ARG5, ARG18)), protectedSqrt(protectedLog(myif(ARG15, 229.5, ARG9, ARG14))), sub(Min2(ARG25, ARG26), ARG26)), protectedDiv(mul(pow2(add(ARG15, ARG23)), Min2(add(ARG1, ARG13), protectedLog(ARG14))), ARG21)))))))))
            
            #Estimator GP-PN
            #estimated_T[k,l] = protectedExp(( min(((min(abs((ARG13 - 229.5)),  max(ARG4, ARG2)) - myif(ARG5, ARG11, ARG10, ARG5))  -  (min(ARG11, min((ARG25 * ARG20), ARG6)) - ARG5)), ARG5) - ARG5 ))
    
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
    degradedScene = np.float64( imread('Images/%s.png' % (imgName) ) )
    degradedScene = degradedScene[:,:,0:3]
    estimatedA = estimateAirlight(degradedScene, 19)
    dehazed_Img = dehazeScene(degradedScene, estimatedA)
    imsave('Results/%s_outputPD.png' % (imgName), np.uint8(dehazed_Img))



        
        
        
    

