# virtual_mouse_all_in_one.py
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import autopy
import pyautogui

# ---------- Hand Detector ----------
class handDetector():
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, 
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList, yList = [], []
        bbox = []
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = img.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20),
                              (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = [0,0,0,0,0]
        if len(self.lmList) == 0:
            return fingers

        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0]-1][1]:
            fingers[0] = 1

        # Other 4 fingers
        for i in range(1,5):
            if self.lmList[self.tipIds[i]][2] < self.lmList[self.tipIds[i]-2][2]:
                fingers[i] = 1

        return fingers

    def findDistance(self, p1, p2, img=None, draw=True, r=8, t=2):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        if draw and img is not None:
            cv2.line(img, (x1,y1), (x2,y2), (255,0,255), t)
            cv2.circle(img,(x1,y1), r, (255,0,255), cv2.FILLED)
            cv2.circle(img,(x2,y2), r, (255,0,255), cv2.FILLED)
            cv2.circle(img,(cx,cy), r, (0,0,255), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)
        return length, img, [x1,y1,x2,y2,cx,cy]


# ---------- Main Program ----------
def main():
    wCam, hCam = 1280, 720
    frameR = 100
    smoothening = 7

    pTime = 0
    plocX, plocY = 0, 0

    drag_active = False

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    detector = handDetector(maxHands=1)

    wScr, hScr = autopy.screen.size()

    print("Virtual Mouse Started -- Press 'q' to quit")

    while True:
        success, img = cap.read()
        if not success:
            continue

        img = detector.findHands(img)
        lmList, _ = detector.findPosition(img)
        fingers = [0,0,0,0,0]

        if lmList:
            fingers = detector.fingersUp()

            xIndex, yIndex = lmList[8][1:]
            xMiddle, yMiddle = lmList[12][1:]
            xThumb,  yThumb  = lmList[4][1:]

            cv2.rectangle(img, (frameR, frameR),
                          (wCam - frameR, hCam - frameR),
                          (255,0,255), 2)

            # ------------------- MOUSE MOVEMENT -------------------
            if fingers[1] == 1 and fingers[2] == 0:
                # Release drag if active
                if drag_active:
                    autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)
                    drag_active = False

                x3 = np.interp(xIndex, (frameR, wCam-frameR), (0, wScr))
                y3 = np.interp(yIndex, (frameR, hCam-frameR), (0, hScr))

                clocX = plocX + (x3 - plocX)/smoothening
                clocY = plocY + (y3 - plocY)/smoothening

                autopy.mouse.move(wScr - clocX, clocY)
                plocX, plocY = clocX, clocY

            # ------------------- CLICK -------------------
            length_click, img, _ = detector.findDistance(8, 12, img)

            if fingers[1] == 1 and fingers[2] == 1 and length_click < 35:
                autopy.mouse.click()
                time.sleep(0.20)

            # ------------------- RIGHT CLICK -------------------
            length_rc, _, _ = detector.findDistance(4, 8, img, draw=False)
            if length_rc < 30:
                autopy.mouse.click(autopy.mouse.Button.RIGHT)
                time.sleep(0.25)

            # ------------------- DRAG -------------------
            if fingers[1] == 1 and fingers[2] == 1 and length_click < 30:
                if not drag_active:
                    autopy.mouse.toggle(autopy.mouse.Button.LEFT, True)
                    drag_active = True
            else:
                if drag_active:
                    autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)
                    drag_active = False

            # ------------------- SCROLL -------------------
            if fingers[1] == 1 and fingers[2] == 1 and length_click > 40:
                dy = yMiddle - yIndex
                pyautogui.scroll(int(-dy * 2))

        # FPS display
        cTime = time.time()
        fps = 1 / (cTime - pTime + 0.0001)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20,50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

        cv2.imshow("Virtual Mouse", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
