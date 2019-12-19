import cv2
import random
import time
from time import sleep
import numpy as np
import imutils

MENU = 0
SINGLEPLAYER = 1
MULTIPLAYER = 2

RASPBERRY = cv2.imread("raspberry.png", -1)
BOMB = cv2.imread("bomb.png", -1)
BANANA = cv2.imread("banana.png", -1)
BURAK = cv2.imread("burak.png", -1)


def CameraInit():
    print("Initializing camera...")
    for i in range(0, 5):
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            print("Camera init OK")
            return camera
        sleep(1)
    print("Camera init failed")
    exit()


def GetImage(camera):
    (ret, frame) = camera.read()

    b_channel, g_channel, r_channel = cv2.split(frame)

    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50  # creating a dummy alpha channel image.

    frame = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    return frame


def GetMaskedFrame(img):
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([160, 100, 120])
    upper_red = np.array([190, 255, 255])

    img_color_mask = cv2.inRange(img_color, lower_red, upper_red)
    img_color = cv2.bitwise_and(img, img, mask=img_color_mask)
    img_grayscale = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_grayscale, (5, 5), 0)
    img_tresh = cv2.threshold(img_blurred, 60, 255, cv2.THRESH_BINARY)[1]

    return img_tresh


def GetPointer(img, ratio):
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    largestArea = 0
    largestContour = []
    for contour in contours:
        thisArea = cv2.contourArea(contour)
        if thisArea > largestArea:
            contour = contour.astype("float")
            contour *= ratio
            contour = contour.astype("int")
            largestContour = contour

    return largestContour


def GetPointerPosition(pointer):
    M = cv2.moments(pointer)

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return (cX, cY)


def MoveFruit(imgFlipped, game):
    fruits = game.fruits

    if len(fruits) == 0 or random.randint(0, 20) == 0:
        fruits.append(Fruit(imgFlipped.shape[1], imgFlipped.shape[0]))

    for fruit in fruits:
        imgFlipped, game = MoveSpecificFruit(fruit, imgFlipped, game)

    return game, imgFlipped


def MoveSpecificFruit(fruit, imgFlipped, game):
    currentTime = int(round(time.time() * 1000))
    fruit.positionX = int(
        round(fruit.startPositionX + ((currentTime - fruit.startTime) / fruit.lifeTime) * (
                fruit.endPositionX - fruit.startPositionX)))
    timePassed = currentTime - fruit.startTime
    fruit.positionY = int(
        round(fruit.startPositionY - (fruit.velocity * timePassed - 30 * timePassed * timePassed / 1000) / 1000))

    # rotated = imutils.rotate_bound(fruitType, fruit.rotateAngle * timePassed / 1000)
    rotated = imutils.rotate(fruit.type, fruit.rotateAngle * timePassed / 1000)

    if imgFlipped.shape[0] - rotated.shape[0] > fruit.positionY > 0:
        y1, y2 = fruit.positionY, fruit.positionY + rotated.shape[0]
        x1, x2 = fruit.positionX, fruit.positionX + rotated.shape[1]
        alpha_s = rotated[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            imgFlipped[y1:y2, x1:x2, c] = (alpha_s * rotated[:, :, c] + alpha_l * imgFlipped[y1:y2, x1:x2, c])
    elif not (imgFlipped.shape[0] - rotated.shape[0] > fruit.positionY > 0) and timePassed > 1000:
        game.fruits.remove(fruit)
        if fruit.type != 1:
            game.lifes = game.lifes - 1

    return imgFlipped, game


def ComputeFruit(game, pointer):
    raspberry = RASPBERRY
    (pointerX, pointerY) = GetPointerPosition(pointer)

    for fruit in game.fruits:
        if fruit.positionY < pointerY < fruit.positionY + raspberry.shape[
            0] and fruit.positionX < pointerX < fruit.positionX + raspberry.shape[0]:
            #TODO change bombbbb
            if fruit.type == BOMB:
                return game, MENU
            game.points = game.points + 1
            game.fruits.remove(fruit)

    return game, SINGLEPLAYER


class Fruit:
    def __init__(self, imageWidth, imageHeight):
        self.startTime = int(round(time.time() * 1000))
        self.lifeTime = random.randint(5, 10) * 1000
        type = random.randint(1, 4)
        if type == 1:
            self.type = BOMB
            fruitWidth, fruitHeight = BOMB.shape[0], BOMB.shape[1]
        elif type == 2:
            self.type = BANANA
            fruitWidth, fruitHeight = BANANA.shape[0], BANANA.shape[1]
        else:
            self.type = RASPBERRY
            fruitWidth, fruitHeight = RASPBERRY.shape[0], RASPBERRY.shape[1]
        self.positionX = random.randint(0, imageWidth - fruitWidth)
        if self.positionX > imageWidth / 2:
            self.endPositionX = random.randint(0, self.positionX)
        else:
            self.endPositionX = random.randint(self.positionX, imageWidth - fruitWidth)

        self.startPositionX = self.positionX

        self.startPositionY = imageHeight - fruitHeight
        self.positionY = self.startPositionY
        maxHeight = random.randint(3 * imageHeight / 4, imageHeight)
        self.velocity = 2000 * maxHeight / self.lifeTime + 20 * self.lifeTime / 2000  # na sekundy

        self.rotateAngle = random.randint(0, 200)


class Game:
    def __init__(self):
        self.lifes = 10
        self.points = 0
        self.fruits = []


def GenerateMenu(imgFlipped, pointer, chosenMode, startTime):
    mode = MENU
    cv2.rectangle(imgFlipped, (100, 100), (200, 200), (0, 255, 0), -1)
    cv2.rectangle(imgFlipped, (300, 300), (400, 400), (0, 255, 0), -1)

    if len(pointer):
        (pointerX, pointerY) = GetPointerPosition(pointer)
        cv2.circle(imgFlipped, (pointerX, pointerY), 7, (255, 255, 255), -1)

        currentMode = -1
        if 100 < pointerX < 200 and 100 < pointerY < 200:
            currentMode = SINGLEPLAYER
        #         if pointerX > 300 and pointerX < 400 and pointerY > 300 and pointerY < 400:
        #             currentMode = MULTIPLAYER
        if currentMode == chosenMode and (startTime + 1000) < int(round(time.time() * 1000)):
            return imgFlipped, chosenMode, None, None, Game()
        elif currentMode != chosenMode and currentMode != -1:
            startTime = int(round(time.time() * 1000))
            return imgFlipped, mode, currentMode, startTime, None
        elif currentMode == -1:
            return imgFlipped, mode, 0, None, None

    return imgFlipped, mode, chosenMode, startTime, None


def GenerateGame(imgFlipped, pointer, game):
    mode = SINGLEPLAYER
    if game.lifes == 0:
        return imgFlipped, MENU, game
    (game, imgFlipped) = MoveFruit(imgFlipped, game)

    if len(pointer):
        (pointerX, pointerY) = GetPointerPosition(pointer)
        (game, mode) = ComputeFruit(game, pointer)
        cv2.circle(imgFlipped, (pointerX, pointerY), 7, (255, 255, 255), -1)

    cv2.putText(imgFlipped, "Wynik: " + str(game.points), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                lineType=cv2.LINE_AA)

    for i in range(0, game.lifes):
        y1, y2 = 5, 5 + BURAK.shape[0]
        x1, x2 = imgFlipped.shape[1] - i * 40 - BURAK.shape[1], imgFlipped.shape[1] - i * 40
        alpha_s = BURAK[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            imgFlipped[y1:y2, x1:x2, c] = (alpha_s * BURAK[:, :, c] + alpha_l * imgFlipped[y1:y2, x1:x2, c])

    return imgFlipped, mode, game


def ProcessFrame():
    camObj = CameraInit()
    debug = True
    capture = True
    mode = MENU
    chosenMode = None
    time = None
    game = None
    while capture:
        img = GetImage(camObj)
        imgFlipped = cv2.flip(img, 1)

        imgResized = cv2.resize(imgFlipped, (270, 200))
        ratio = img.shape[0] / float(imgResized.shape[0])

        imgMasked = GetMaskedFrame(imgResized)
        pointer = GetPointer(imgMasked, ratio)

        if mode == MENU:
            (gameImage, mode, chosenMode, time, game) = GenerateMenu(imgFlipped, pointer, chosenMode, time)

        elif mode == SINGLEPLAYER:
            (gameImage, mode, game) = GenerateGame(imgFlipped, pointer, game)

        elif mode == MULTIPLAYER:
            capture = False

        if cv2.waitKey(1) == ord('q'):
            capture = False

        if debug: cv2.imshow("Image preview", imgMasked)
        cv2.imshow("Game preview", gameImage)


ProcessFrame()
