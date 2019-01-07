import wx
import numpy as np

from .wxgraphics import *


__all__ = ['Pong',]


class PongObject:
    def __init__(self):
        self.coords = np.array([0.0,0.0])
        self.velocity = np.array([0.0,0.0])
   
        self.pen = wx.Pen((250,250,250), 2)
        self.brush = wx.Brush((255,255,255))

    def outsideOfScreen(self, windowDiameter):
        return False
    
    def update(self, windowDiameter):
        self.coords += self.velocity 
  
    def setCoords(self, coords):
        self.coords[:] = coords
    
    def setXCoord(self, x):
        self.coords[0] = x

    def setYCoord(self, y):
        self.coords[1] = y

    def setVelocity(self, velocity):
        self.velocity[:] = velocity

class Ball(PongObject):
    def __init__(self):
        PongObject.__init__(self)

        self.diameterPercentage = 0.05

        #self.speedPercentage = 0.01
        self.speedPercentage = 0.004

        self.direction = 2 

        self.touchingBottom = False

    def draw(self, gc, windowDiameter):
        gc.PushState()
        gc.Translate(*(self.coords*windowDiameter))

        gc.SetPen(self.pen)
        gc.SetBrush(self.brush)

        gc.DrawEllipse(0.0, 0.0, self.diameterPercentage * windowDiameter, 
                                 self.diameterPercentage * windowDiameter)

        gc.PopState()

    def bounceOfWalls(self):
        if self.coords[0] + self.diameterPercentage > 0.995: 
            self.reflectVertical()
            
        if self.coords[0] < 0.005:
            self.reflectVertical() 

        if self.coords[1] < 0.005:
            self.reflectHorizontal()
            
        if self.coords[1] + self.diameterPercentage > 0.995:
            self.touchingBottom = True
        
        return False  

    def reflectVertical(self):
        self.velocity[0] *= -1

    def reflectHorizontal(self):
        self.velocity[1] *= -1

    def update(self, windowDiameter):
        if self.touchingBottom:
            return

        self.bounceOfWalls()
        PongObject.update(self, windowDiameter)

    def setSpeed(self, speedPercentage):
        self.speedPercentage = speedPercentage

    def squareDistance(self, pointA, pointB):
        xA, yA = pointA[0], pointA[1]
        xB, yB = pointB[0], pointB[1]
        return (xA - xB)**2 + (yA - yB)**2 

    def intersectCircle(self, pointA, pointB):
        circleCenter = self.coords + self.diameterPercentage*0.5

        lineLength = (self.squareDistance(pointA, pointB))**0.5

        circleCenter[0] -= pointA[0]
        circleCenter[1] -= pointA[1]
        
        pointB[0] -= pointA[0]
        pointB[1] -= pointA[1]

        pointB[0] /= lineLength
        pointB[1] /= lineLength

        projectionPoint = [pointB[0]*circleCenter[0],pointB[1]*circleCenter[1]]
        distanceToLine = (self.squareDistance([0,0], circleCenter) - self.squareDistance([0,0], projectionPoint))**0.5

        if distanceToLine <= self.diameterPercentage/2:
            return True
        else:
            return False

    def dealWithPaddle(self, paddle):
        """
        pointA = [paddle.coords[0], paddle.coords[1]]
        pointB = [paddle.coords[0] + paddle.widthPercentage, paddle.coords[1]] 
        pointC = [paddle.coords[0] + paddle.widthPercentage, paddle.coords[1] + paddle.heightPercentage]
        pointD = [paddle.coords[0], paddle.coords[1] + paddle.heightPercentage]

        circleCenter = [self.coords[0] + self.diameterPercentage*0.5, 
                        self.coords[1] + self.diameterPercentage*0.5] 

        circleRadius = self.diameterPercentage/2

        if self.squareDistance(circleCenter, pointA) <= circleRadius**2:
            # add ability to reflect at a different angle
            self.reflectHorizontal()
            return True

        if self.squareDistance(circleCenter, pointB) <= circleRadius**2:
            # add ability to reflect at a different angle
            self.reflectHorizontal()
            return True

        if circleCenter[0] >= pointA[0] and circleCenter[0] <= pointB[0]:
            if self.intersectCircle(pointA, pointB):
                self.coords[1] = pointA[1] - self.diameterPercentage
                self.reflectHorizontal()
                return True

        if circleCenter[0] >= pointD[0] and circleCenter[0] <= pointC[0]:
            if self.intersectCircle(pointD, pointC):
                self.coords[1] = pointD[1]
                self.reflectHorizontal()  
                return True

        if circleCenter[1] >= pointA[1] and circleCenter[0] <= pointD[1]:
            if self.intersectCircle(pointD, pointA):
                self.coords[0] = pointD[0] - self.diameterPercentage
                self.reflectVertical()
                return True

        if circleCenter[1] >= pointB[1] and circleCenter[0] <= pointC[1]:
            if self.intersectCircle(pointC, pointB):
                self.coords[0] = pointC[0] 
                self.reflectVertical()
                return True

        return False
        """
        paddleLeft = paddle.coords
        paddleRight = paddle.coords + np.array([paddle.widthPercentage, 0])

        ballLeft = self.coords + np.array([0, self.diameterPercentage])
        ballRight = self.coords + np.array([self.diameterPercentage, self.diameterPercentage])

        if paddleLeft[0] < ballLeft[0] and paddleRight[0] > ballRight[0]:
            if paddleLeft[1] < ballLeft[1]:
                self.reflectHorizontal()
                self.coords[1] = paddleLeft[1] - 1.0001*self.diameterPercentage
                return True

        return False

    def setRadius(self, radiusPercentage):
        self.radiusPercentage = radiusPercentage

class Paddle(PongObject):
    def __init__(self, widthPercentage = 0.4, heightPercentage = 0.025):
        PongObject.__init__(self)
        self.widthPercentage = widthPercentage 
        self.heightPercentage = heightPercentage

        #self.speedPercentage = 0.015
        self.speedPercentage = 0.005

    def draw(self, gc, windowDiameter):
        gc.PushState()
        gc.Translate(*(self.coords*windowDiameter))

        gc.SetPen(self.pen)
        gc.SetBrush(self.brush)

        gc.DrawRectangle(0.0, 0.0,
                self.widthPercentage*windowDiameter,
                self.heightPercentage*windowDiameter)

        gc.PopState()

    def moveLeft(self):
        self.setVelocity(np.array([-self.speedPercentage,0.0]))

    def moveRight(self):
        self.setVelocity(np.array([self.speedPercentage,0.0]))

    def stopMoving(self):
        self.setVelocity(np.array([0.0,0.0]))

    def update(self, windowDiameter):
        self.setYCoord(0.95-self.heightPercentage)
        PongObject.update(self, windowDiameter)  

        if self.coords[0] < 0.005:
            self.coords[0] = 0.005

        if self.coords[0] + self.widthPercentage > 0.995:
            self.coords[0] = 0.995 - self.widthPercentage

    def setWidth(self, widthPercentage):
        self.widthPercentaga = widthPercentage
  
    def setHeight(self, heightPercentage):
        self.heightPercentage - heightPercentage

    def setSpeed(self, speedPercentage):
        self.speedPercentage = speedPercentage

class Border(PongObject):
    def __init__(self, widthPercentage = 0.4, heightPercentage = 0.05, coords=np.array([0.0,0.0])):
        PongObject.__init__(self)
        self.widthPercentage = widthPercentage 
        self.heightPercentage = heightPercentage
        self.setCoords(coords)

    def draw(self, gc, windowDiameter):
        gc.PushState()
        gc.Translate(*(self.coords*windowDiameter))

        gc.SetPen(self.pen)
        gc.SetBrush(self.brush)

        gc.DrawRectangle(0.0, 0.0, self.widthPercentage*windowDiameter, self.heightPercentage*windowDiameter)

        gc.PopState()

class Scoreboard(PongObject):
    def __init__(self):
        PongObject.__init__(self)

        self.score = np.array([0,0])

        self.scoreFont = wx.Font(pointSize=12, family=wx.FONTFAMILY_SWISS,
                style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL, underline=False)

        self.setCoords(np.array([0.07, 0.06]))

    def scoreGood(self):
        self.score[0] += 1

    def scoreBad(self):
        self.score[1] += 1
      
    def reset(self):
        self.score[...] = 0

    def draw(self, gc, windowDiameter):
        gc.PushState()
        gc.Translate(*(self.coords*windowDiameter))

        gc.SetPen(self.pen)
        gc.SetBrush(self.brush)

        gc.Scale(0.008*windowDiameter, 0.008*windowDiameter)

        gc.SetFont(gc.CreateFont(self.scoreFont, col='white'))
        gc.DrawText(str(self.score[0]) + ":" + str(self.score[1]),0,0)

        gc.PopState()     

    def update(self, windowDiameter):
        pass

class Pong(GraphicsPanel):
    def __init__(self, *args, **kwargs):
        self.initGame()

        GraphicsPanel.__init__(self, *args, **kwargs)

        self.initRefreshTimer()

    def initGame(self):
        self.ball = Ball()
        self.paddle = Paddle()
        self.scoreboard = Scoreboard()

        self.topBorder = Border(1,0.01, np.array([0,0]))
        self.rightBorder = Border(0.01, 1, np.array([0,0]))
        self.leftBorder = Border(0.01, 1, np.array([0.99,0]))

        self.gameObjects = [self.ball, self.paddle, self.scoreboard,
                            self.topBorder, self.rightBorder, self.leftBorder]

        self.isPlaying = False

    def initRefreshTimer(self):
        self.refreshTimer = wx.Timer(self) # timer to update game state
        self.Bind(wx.EVT_TIMER, self.update, self.refreshTimer) # handle timer events

        self.bindKeyboard()

        self.refreshTimer.Start(30.0)

    def bindKeyboard(self):
        self.Bind(wx.EVT_KEY_DOWN, self.onKeyPress)    
        self.Bind(wx.EVT_KEY_UP, self.onKeyRelease)

    def onKeyPress(self, event):
        keycode = event.GetKeyCode()

        if keycode == ord('A') or keycode == ord('a') or keycode == wx.WXK_LEFT:
            self.movePaddleLeft()

        if keycode == ord('D') or keycode == ord('d') or keycode == wx.WXK_RIGHT:
            self.movePaddleRight()

        if keycode == ord('R') or keycode == ord('r'):
            self.startGame()
    
    def onKeyRelease(self, event):
        self.stopPaddle()

    def getScore(self):
        return self.scoreboard.score

    def newGame(self):
        self.scoreboard.reset()
        self.startGame()

    def startGame(self):
        self.isPlaying = True
        self.ball.touchingBottom = False
        self.ball.setCoords(np.array([0.5,0.5]))

        direction = np.random.uniform(-0.4*np.pi, 0.4*np.pi)
        velocity = self.ball.speedPercentage * -np.array([np.sin(direction), np.cos(direction)])

        self.ball.setVelocity(velocity)

    def stopGame(self):
        self.isPlaying = False

    def movePaddleLeft(self):
        self.paddle.moveLeft()

    def movePaddleRight(self):
        self.paddle.moveRight()

    def stopPaddle(self):
        self.paddle.stopMoving()

    def update(self, event=None):
        windowDiameter = self.winRadius*2

        if self.isPlaying:
          if self.ball.touchingBottom:
              self.scoreboard.scoreBad()
              self.startGame()

          if self.ball.dealWithPaddle(self.paddle):
             self.scoreboard.scoreGood()

          [obj.update(windowDiameter) for obj in self.gameObjects]

        self.refresh()

    def draw(self, gc):
        # Make the game centered
        gc.PushState()
        gc.Translate((self.GetSize()[0] - self.winRadius*2)*0.5, 0)
 
        [obj.draw(gc, self.winRadius*2) for obj in self.gameObjects]

        gc.PopState()


if __name__ == '__main__':
    class PongFrame(wx.Frame):
        def __init__(self):
            wx.Frame.__init__(self, parent=None, title='Pong')

            self.sizer = wx.BoxSizer(orient=wx.VERTICAL)

            self.pongPanel = Pong(self)
            self.sizer.Add(self.pongPanel, proportion=1, flag=wx.EXPAND)

            self.SetSizer(self.sizer)

    class PongApp(wx.App):
        def OnInit(self):
            self.SetAppName('Pong')
            pongFrame = PongFrame()
            pongFrame.Show()
            pongFrame.pongPanel.startGame()
            return True

    app = PongApp()
    app.MainLoop()
