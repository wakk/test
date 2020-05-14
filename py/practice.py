# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
"""
img=cv2.imread(r'mei.jpg',0)
cv2.namedWindow('title', cv2.WINDOW_NORMAL)
cv2.imshow('title', img)
k=cv2.waitKey(0)
if k==ord('s'):#ASC码转换
    cv2.imwrite("lena_save.bmp", img)
cv2.destroyAllWindows()
"""
"""
img=cv2.imread(r'mei.jpg',0)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(122)
plt.xticks([]), plt.yticks([]) # 隐藏x和y轴
plt.imshow(img2)
plt.show()
"""
"""
capture = cv2.VideoCapture(0)
# 定义编码方式并创建VideoWriter对象
fourcc=cv2.VideoWriter_fourcc(*'MJPG')
outfile=cv2.VideoWriter('output.avi',fourcc,25.,(640,480))
while(capture.isOpened()):
    ret,frame=capture.read()
    if ret:
        outfile.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1)==ord('q'):
            break
capture.release()
cv2.destroyAllWindows()
"""
"""
capture=cv2.VideoCapture(r'E:\car.mp4')
while(capture.isOpened()):
    ret,frame=capture.read()
    if ret==False:
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)# 将这帧转换为灰度图
    cv2.imshow('frame',gray)
    print(cv2.CAP_PROP_POS_FRAMES)
    print(cv2.CAP_PROP_FRAME_COUNT)
    if cv2.waitKey(30)==ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
"""
"""
def nothing(x):
    pass
capture=cv2.VideoCapture(r'E:\car.mp4')
frame_num=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))#7
cv2.namedWindow('play')
cv2.createTrackbar('P', 'play', 0, frame_num,nothing)
p=cv2.getTrackbarPos('P','play')
print(p)
while(capture.isOpened()):
    ret,frame=capture.read()
    if ret==False:
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)# 将这帧转换为灰度图
    cv2.imshow('play',gray)
    if cv2.waitKey(1)==ord('q'):
        break
    p=cv2.getTrackbarPos('P','play')
    capture.set(cv2.CAP_PROP_POS_FRAMES,p)  #设置要获取的帧号
capture.release()
cv2.destroyAllWindows()
"""
"""
img=cv2.imread('meimei.jpg')
px=img[100,90]
print(px)
# 只获取蓝色blue通道的值
px_blue=img[100,90,0]
print(px_blue)
img[100, 90] = [255, 255, 255]
print(img[100, 90])  # [255 255 255]
print(img.dtype)
b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))
b = img[:, :, 0]#numpy获取B通道
cv2.imshow('blue', b)
cv2.waitKey(0)
cv2.namedWindow('title', cv2.WINDOW_NORMAL)
cv2.imshow('title',r)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
#获取蓝色上下限
blue = np.uint8([[[255, 0, 0]]])
hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
print(hsv_blue)  # [[[120 255 255]]]
#提取绿色
capture=cv2.VideoCapture(r'color.mp4')
lower_blue=np.array([50,110,110])
upper_blue=np.array([80,255,255])
cv2.namedWindow("mask", 0)  # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
cv2.resizeWindow("mask", 800, 500)    # 设置长和宽
cv2.namedWindow("res", 0)  # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
cv2.resizeWindow("res", 800,500)    # 设置长和宽
while(True):
    # 1.捕获视频中的一帧
    ret, frame = capture.read()
    if ret==False:
        break
    # 2.从BGR转换到HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 3.inRange()：介于lower/upper之间的为白色，其余黑色
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 4.只保留原图中的蓝色部分
    res = cv2.bitwise_and(frame, frame, mask=mask)

    #cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    if cv2.waitKey(1) == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
"""
"""
bs = cv2.createBackgroundSubtractorKNN(detectShadows = True)
camera = cv2.VideoCapture(r'E:\car.mp4')
 
while True:
	ret, frame = camera.read()
	fgmask = bs.apply(frame)
	fg2 = fgmask.copy()
	th = cv2.threshold(fg2,244,255,cv2.THRESH_BINARY)[1]
	dilated = cv2.dilate(th,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations = 2)
	contours, hier = cv2.findContours(dilated,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for c in contours:
		if cv2.contourArea(c) > 100:
			(x,y,w,h) = cv2.boundingRect(c)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
			
	cv2.imshow("mog",fgmask)
	cv2.imshow("thresh",th)
	cv2.imshow("detection",frame)
	if cv2.waitKey(24) == ord('q'):
		break
camera.release()
cv2.destroyAllWindows()
"""
"""
#灰度图读入
img1=cv2.imread('meimei.jpg')
img=cv2.imread('meimei.jpg',0)
gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)# 将这帧转换为灰度图
#阈值分割
ret,th=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow('thresh',th)
cv2.imshow('gray',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 应用5种不同的阈值方法
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
titles = ['Original', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, th1, th2, th3, th4, th5]
# 使用Matplotlib显示
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i], fontsize=8)
    plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
plt.show()
"""
"""
# 自适应阈值对比固定阈值
img=cv2.imread('meimei.jpg',0)
#固定阈值
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#自适应阈值
th2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                          cv2.THRESH_BINARY,11,8)
th3=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            cv2.THRESH_BINARY, 7, 15)
titles = ['Original', 'Global(v = 127)', 'Adaptive Mean', 'Adaptive Gaussian']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i], fontsize=8)
    plt.xticks([]), plt.yticks([])
plt.imshow(images[2], 'gray')
plt.show()
"""
"""
img = cv2.imread('meimei.jpg', 0)
# 固定阈值法
ret1, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
# Otsu阈值法
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 先进行高斯滤波，再使用Otsu阈值法
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
images = [img, 0, th1, img, 0, th2, blur, 0, th3]
titles = ['Original', 'Histogram', 'Global(v=100)',
          'Original', 'Histogram', "Otsu's",
          'Gaussian filtered Image', 'Histogram', "Otsu's"]
for i in range(3):
    # 绘制原图
    plt.subplot(3, 3, i * 3 + 1)
    plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3], fontsize=8)
    plt.xticks([]), plt.yticks([])

    # 绘制直方图plt.hist，ravel函数将数组降成一维
    plt.subplot(3, 3, i * 3 + 2)
    plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1], fontsize=8)
    plt.xticks([]), plt.yticks([])

    # 绘制阈值图
    plt.subplot(3, 3, i * 3 + 3)
    plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2], fontsize=8)
    plt.xticks([]), plt.yticks([])
plt.show()
"""
"""
img=cv2.imread('meimei.jpg')
rows, cols = img.shape[:2]
# 定义平移矩阵，需要是numpy的float32类型
# x轴平移100，y轴平移50
M = np.float32([[1, 0, 100], [0, 1, 50]])
res=cv2.resize(img,(132,150))#放缩
res = cv2.flip(res, 1)#水平翻转
dst = cv2.warpAffine(res, M, (cols, rows))# 用仿射变换实现平移
res2=cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
res2=cv2.flip(res2,0)
# 45°旋转图片并缩小一半
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.5)
dst2 = cv2.warpAffine(res2, M, (cols, rows))
cv2.imshow('shrink', dst), cv2.imshow('zoom', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
img = cv2.imread('card.jpg')
# 原图中卡片的四个角点,以后可以特征提取直接找点
pts1 = np.float32([[148, 80], [437, 114], [94, 247], [423, 288]])
# 变换后分别在左上、右上、左下、右下四个点
pts2 = np.float32([[0, 0], [320, 0], [0, 178], [320, 178]])
# 生成透视变换矩阵
M = cv2.getPerspectiveTransform(pts1, pts2)
# 进行透视变换，参数3是目标图像大小
dst = cv2.warpPerspective(img, M, (320, 178))
plt.subplot(121), plt.imshow(img[:, :, ::-1]), plt.title('input')
plt.subplot(122), plt.imshow(dst[:, :, ::-1]), plt.title('output')
plt.show()
"""
"""
# 创建一副黑色的图片
img = np.zeros((512, 512, 3), np.uint8)
# 画一条线宽为5的蓝色直线，参数2：起点，参数3：终点
cv2.line(img, (0, 0), (512, 512), (255, 0, 255), 5)
# 画一个绿色边框的矩形，参数2：左上角坐标，参数3：右下角坐标
cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
# 画一个填充红色的圆，参数2：圆心坐标，参数3：半径
cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)
# 在图中心画一个填充的半圆
cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, (255, 0, 0), -1)
# 定义四个顶点坐标
pts = np.array([[10, 5],  [50, 10], [70, 20], [20, 30]], np.int32)
# 顶点个数：4，矩阵变成4*1*2维
pts = pts.reshape((-1, 1, 2))
cv2.polylines(img, [pts], True, (0, 255, 255))
# 添加文字
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'ex2tron', (10, 500), font,
            4, (255, 255, 255), 2, lineType=cv2.LINE_AA)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
drawing = False  # 是否开始画图
mode = True  # True：画矩形，False：画圆
start = (-1, -1)
# 获取所有的事件
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)
def mouse_event(event, x, y, flags, param):
    global start, drawing, mode

    # 左键按下：开始画图
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start = (x, y)
        global last
        last=(x,y)
    # 鼠标移动，画图
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                if start!=last:
                    cv2.rectangle(img, start, last, (0, 0, 0), 1)
                cv2.rectangle(img, start, (x, y), (0, 255, 0), 1)
                last=(x, y)
            else:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    # 左键释放：结束画图
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.rectangle(img, start, (x, y), (0, 255, 0), 1)
        else:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_event)
while(True):
    cv2.imshow('image', img)
    # 按下m切换模式
    if cv2.waitKey(1) == ord('m'):
        mode = not mode
    elif cv2.waitKey(1) ==ord('q'):
        break
cv2.destroyAllWindows()
"""
"""
img1 = cv2.imread('meimei.jpg')
img2 = cv2.imread('cv.png')
# 把logo放在左上角，所以我们只关心这一块区域
rows, cols = img2.shape[:2]
roi = img1[:rows, :cols]
# 创建掩膜
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
mask_inv2= cv2.bitwise_not(mask_inv)
# 保留除logo外的背景
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv2)
cv2.imshow('tt',img1_bg )
img2_bg = cv2.bitwise_and(img2, img2, mask=mask_inv)
dst = cv2.add(img1_bg,img2_bg)  # 进行融合
cv2.imshow('tt1',img2_bg)
img1[:rows, :cols] = dst  # 融合后放在原图上
cv2.imshow('tt3',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('mei.jpg')
# 此处需注意，请参考后面的解释
res = np.uint8(np.clip((1.8 * img + 10), 0, 255))
tmp = np.hstack((img, res))  # 两张图片横向合并（便于对比显示）
cv2.imshow('image', tmp)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
img = cv2.imread('mei.jpg')
blur = cv2.blur(img, (3, 3))  # 均值模糊
# 前面的均值滤波也可以用方框滤波实现：normalize=True
blur = cv2.boxFilter(img, -1, (3, 3), normalize=True)
# 均值滤波vs高斯滤波
blur = cv2.blur(img, (5, 5))  # 均值滤波
gaussian = cv2.GaussianBlur(img, (5, 5), 1)  # 高斯滤波
blur = cv2.bilateralFilter(img, 9, 75, 75)  # 双边滤波
median = cv2.medianBlur(img, 5)  # 中值滤波
tmp = np.hstack((gaussian,blur,img))
cv2.imshow('image1',tmp)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('meimei.jpg')
# 定义卷积核
kernel = np.ones((3, 3), np.float32) / 10
default = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)#默认扩展
# 卷积操作，-1表示通道数与原图相同
dst = cv2.filter2D(default , -1, kernel)
cv2.imshow('image',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
img = cv2.imread('meimei.jpg', 0)
gaussian = cv2.GaussianBlur(img, (5, 5), 1)  # 高斯滤波
#_, thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('canny1', gaussian)
edges = cv2.Canny(gaussian, 30, 70)  # canny边缘检测
cv2.imshow('canny', np.hstack((img, edges)))
cv2.waitKey(0)
cv2.destroyAllWindows()

# 回调函数，x表示滑块的位置，本例暂不使用
def nothing(x):
    pass
img =cv2.imread('mei.jpg', 0)
cv2.namedWindow('image')
# 创建RGB三个滑动条
cv2.createTrackbar('MAX', 'image', 37, 255, nothing)
cv2.createTrackbar('MIN', 'image', 0, 255, nothing)
img2=img
while(True):
    cv2.imshow('image', img2)
    if cv2.waitKey(1) == 27:
        break

    # 获取滑块的值
    maxv = cv2.getTrackbarPos('MAX', 'image')
    minv = cv2.getTrackbarPos('MIN', 'image')
    # 设定img的颜色
    _, thresh = cv2.threshold(img, minv,maxv, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img2=cv2.Canny(thresh, 30, 70)  # canny边缘检测
cv2.destroyAllWindows()
"""
"""
img=cv2.imread('meimei.jpg',0)
# 自己进行垂直边缘提取
kernel=np.array([[-1,0,1],
                 [-2,0,2],
                 [-1,0,1]],dtype=np.float32)
dst_v=cv2.filter2D(img,-1,kernel)
# 自己进行水平边缘提取
dst_h = cv2.filter2D(img, -1, kernel.T)
# 横向并排对比显示
sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)  # 只计算x方向
sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)  # 只计算y方向
laplacian = cv2.Laplacian(img, -1)  # 使用Laplacian算子
cv2.imshow('edge',np.hstack((img,dst_v,dst_h)))
cv2.imshow('edge2',np.hstack((img,laplacian)))
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素
img = cv2.imread('mei.jpg', 0)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)#形态学梯度
cv2.imshow('title',np.hstack((opening,closing )))
cv2.imshow('title2',gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
img=cv2.imread('car1.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# 寻找二值化图中的轮廓
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))  # 结果应该为2
r=cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
cnt = contours[1]
draw=cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2)
cv2.imshow('title',r)
cv2.imshow('title2',draw)
cv2.waitKey(0)
cv2.destroyAllWindows()

img=cv2.imread('round3.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours), hierarchy, sep='\n')
c=[]
c.append(contours[2])
c.append(contours[4])
c.append(contours[6])
draw=cv2.drawContours(img, c, -1,(180,215,215), -1)
cv2.imshow('title2',draw)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
img = cv2.imread('31.png', 0)
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh, 3, 2)
# 以数字3的轮廓为例
cnt = contours[2]
img_color1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_color2 = np.copy(img_color1)
cv2.drawContours(img_color1, [cnt], 0, (0, 0, 255), 2)
area = cv2.contourArea(cnt)
perimeter = cv2.arcLength(cnt, True)#周长
M = cv2.moments(cnt)
cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']#质心
x, y, w, h = cv2.boundingRect(cnt)  # 外接矩形
cv2.rectangle(img_color1, (x, y), (x + w, y + h), (0, 255, 0), 2)
rect = cv2.minAreaRect(cnt)  # 最小外接矩形
box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
cv2.drawContours(img_color1, [box], 0, (255, 0, 0), 2)
(x, y), radius = cv2.minEnclosingCircle(cnt)
(x, y, radius) = np.int0((x, y, radius))  # 圆心和半径取整
cv2.circle(img_color2, (x, y), radius, (0, 0, 255), 2)
ellipse = cv2.fitEllipse(cnt)
cv2.ellipse(img_color2, ellipse, (0, 255, 0), 2)
cv2.imshow('title2',np.hstack((img_color1,img_color2)))
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
img1=cv2.imread('mei.jpg',0)
plt.hist(img1.ravel(), 256, [0, 256])
plt.show()
#模糊掉对比度和亮度
equ = cv2.equalizeHist(img1)
# 自适应均衡化，参数可选
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img1)
cv2.imshow('equalization', np.hstack((cl1, equ)))  # 并排显示
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# 1.读入原图和模板,适用于一模一样的识别
img_rgb = cv2.imread('mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('mario_coin.jpg', 0)
h, w = template.shape[:2]
# 2.标准相关模板匹配
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8 

# 3.这边是Python/Numpy的知识，后面解释
loc = np.where(res >= threshold)  # 匹配程度大于%80的坐标y,x
for pt in zip(*loc[::-1]):  # *号表示可选参数
    right_bottom = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, right_bottom, (0,255, 0),1)
cv2.imshow('title',img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()





















 






































