import cv2
import numpy as np
import math
import HandTracking as ht

#---------- To Controll Volume ------------#
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
############################################


##################################
# wCam, hCam = 640, 480#1280, 720
wCam, hCam = 1280, 720
##################################


#---------- To Controll Volume ------------#
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

#----- Volume Scale Converter
# Functions to properly convert volume values between logarithmic and linear scales.
# fiting was made separately from a careful sampling of volume values to obtain proper parameters.
def volPc(logvalue):
    y0 = -1.38577
    A = 101.38577
    R0 = 0.06579
    volPC = int(round((y0 + A*math.exp(R0*logvalue)),1))
    return volPC

def logvolPc(value):
    y0 = -1.38577
    A = 101.38577
    R0 = 0.06579
    logvolPC = (math.log(value-y0)-math.log(A))/R0
    return logvolPC
#-----------------------------------

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = ht.handDetector(detectionCon=0.7)## using the handDetector module
vol = volPc(volume.GetMasterVolumeLevel())

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)#draw=False # drawing landmarks of the hand
    
    lmList = detector.findPosition(img)
    detector.drawCircleFinger(img, lmList, 4)# Thumb tip
    detector.drawCircleFinger(img, lmList, 8)# Index finger tip

    ## modifying the volume according to the distance between the index finger and the thumb
    if len(lmList) != 0:# if == 0, no hands detected.
        length, cx, cy = detector.drawLineTwoFingers(img, lmList, 4, 8)
        #Hand range from 25 to 225 aprox. 
        #Volume range from -65.25 to 0.0, logarithmic
        vol = np.interp(length,[25,225], [0, 100])# linear interpolation 
        volume.SetMasterVolumeLevel(logvolPc(vol), None)# correct logarithmic volume value assignment
        
        if length<25:
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        elif length>225:
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
        else:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        
        if 600<lmList[8][1]<680 and lmList[8][2] < 40:# Exit condition 1
            break
    
    ## setting the volume bar      
    level = np.interp(vol,[0,100], [450,100])
    
    cv2.rectangle(img, (20,100), (40, 450), (0,0,0), 2)
    cv2.rectangle(img, (20,int(level)), (40, 450), (0,0,0), cv2.FILLED)
    
    cv2.putText(img,"Vol." , (10,55), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 2)
    cv2.putText(img,"min", (5,480), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 2)
    cv2.putText(img,"max", (5,85), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 2)
    
    levelperc = int((450 - level)*100 / 350)
    cv2.putText(img,str(levelperc), (45,int(level+10)), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 2)
 
    # --------   Exit condition 1
    # Pointing to the "exit" region of the screen with an index finger
    cv2.rectangle(img, (600,0), (680, 40), (0,0,255), cv2.FILLED)
    cv2.putText(img,"Exit", (607,30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
    
    if detector.handsQuantity()>1:
        lmList_1 = detector.findPosition(img, 1)
        if len(lmList_1) != 0:
            detector.drawCircleFinger(img, lmList_1, 8)
            if 600<lmList_1[8][1]<680 and lmList_1[8][2] < 40:
                break
    #-----------------------------------
            
    cv2.imshow("Cam_Image", img)
    
    # --------   Exit condition 2
    # Pressing the 'Q' key
    if cv2.waitKey(1) == ord('q'):
        break
    #-----------------------------------
    
cap.release()
cv2.destroyAllWindows()