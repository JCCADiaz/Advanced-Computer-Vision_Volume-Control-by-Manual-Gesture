import cv2
import mediapipe as mp
import math

class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon=0.5, trackingCon=0.5):
        ## default values: static_image_mode=False, max_num_hands=2,min_detection_confidence=0.5, min_tracking_confidence=0.5
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
        
    def handsQuantity(self):
        myHands = self.results.multi_hand_landmarks
        if myHands==None:
            return 0
        else:
            return len(myHands)

    def findPosition(self, img, handNo=0):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                
        return lmList
    
    def drawCircleFinger(self, img, lmList, index):
        if len(lmList) != 0:
            cv2.circle(img, (lmList[index][1], lmList[index][2]), 10, (0, 255, 0), cv2.FILLED)
    
    def drawLineTwoFingers(self, img, lmList, index1, index2):
        if len(lmList) != 0:
            cv2.line(img, (lmList[index1][1], lmList[index1][2]), (lmList[index2][1], lmList[index2][2]), (0, 255, 0), 3)
            return (math.hypot(lmList[index2][1]-lmList[index1][1], lmList[index2][2]-lmList[index1][2]), 
                    (lmList[index1][1]+lmList[index2][1])//2 , 
                    (lmList[index1][2]+lmList[index2][2])//2)
