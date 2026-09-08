# 导包
import numpy as np
import argparse
import cv2

#创建参数解析对象
ap = argparse.ArgumentParser()
# 添加命令行参数
ap.add_argument("-i", "--image", required=True, help="path to input image")
#把命令行传入参数转为字典args["image"]就能拿到图片路径
args = vars(ap.parse_args())

def order_points(pts):
    """
	功能：把4个无序的四边形顶点，按【左上、右上、右下、左下】顺序整理好
	pts: shape=(4,2) 的数组，4个点(x,y)，但是顺序是乱的
	return rect: 排好顺序的4个点 [tl, tr, br, bl]
	"""
    #初始化四行二列等等数组 用于存放排好序的四个坐标点，float32类型
    rect = np.zeros((4, 2), dtype="float32")
    #每一个点x+y的和：左上角总和最小，右下角总和最大
    s = pts.sum(axis=1) #axis = 1：按行求和，每个点(x+y)
    rect[0] = pts[np.argmin(s)] # np.argmin(s) 取总和最小索引 -》 左上tl
    rect[2] = pts[np.argmax(s)] # np.argmax(s) 取总和最打索引 -》 右下br

    #np.diff 按行做后列-前列：x2-x1,右上差值最小，左下差值最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # 右上 tr
    rect[3] = pts[np.argmax (diff)] # 左下 bl

    return rect

def four_point_transform(image, pts):
    """
    	四点透视变换：把倾斜的四边形矫正成正矩形（扫描文档拉直）
    	image：原始图片
    	pts：检测得到的文档4个角点（乱序）
    	return warped：矫正完成后的俯视文档图片
    	"""
    # 将输入四个点排序 得到左上、右上、右下、左下
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    #计算输出图片宽度
    # 计算底边两点br-bl的欧氏距离
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    # 计算定边两点tr -tl的欧氏距离
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    #计算图片输出高度
    heightA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    # 计算定边两点tr -tl的欧氏距离
    heightB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    #目标平面坐标，矫正后理想矩形四个角
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # 计算透视变换矩阵M：原始四边形坐标 -》 目标矩形坐标
    M = cv2.getPerspectiveTransform(rect, dst)
    #执行透视变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    #返回矫正完毕的图片
    return warped

def resize(image, width = None, height =None, inter = cv2.INTER_AREA):
    """
    	图片缩放函数，保持宽高比缩放
    	image:输入图像
    	width：目标宽度，height：目标高度，只传一个就自动算另一个
    	inter：插值方式，INTER_AREA适合缩小图片
    	return resized：缩放后的图片
    	"""
    # 获取原图高、宽
    dim = None
    (h, w) = image.shape[:2]
    #如果宽高都不给，直接返回原图，不缩放
    if width is None and height is None:
        return image
    # 如果只给高度，计算缩放比例r，算出对应宽度
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    #如果只给宽度，计算缩放比例r， 算出对应高度
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    # 执行缩放
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

# 主程序开始执行
# 读取传入的图片，cv2.imread读取格式BGR
image = cv2.imread(args["image"])

# 我们后面会把图片缩放到高度500做轮廓检测，但是原始坐标是原图尺寸，所以记录缩放比例ratio
ratio = image.shape[0] / 500.0
orig = image.copy()#备份原始图片，后面变换要用原图，不能用缩放后的小图

#将原图等比例缩放到高度500，方便后续轮廓运算，加快速度
image = resize(orig, height = 500)

#step1:预处理：灰度 高斯模糊，canny边缘检测
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #BGR 彩色图转灰度图
gray = cv2.GaussianBlur(gray, (5, 5), 0) #高斯模糊降噪，消除四岁噪声，5X5卷积核
edged = cv2.Canny(gray, 75, 200) #Canny边缘检测，提取物理轮廓边缘

# 窗口展示原图和边缘检测结果：waitKey(0) 等待任意键继续
print("STEP1: 边缘检测")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()


#step2:轮廓检测，寻找文档四边形
#findContours：寻找所有轮廓；RETR_LIST全部轮廓；CHAIN_APPROX_SIMPLE压缩轮廓点
# OpenCV版本差异：部分版本返回(图像,轮廓,层次)，取索引[1]拿到轮廓列表
cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)#[1]
if len(cnts) == 0:
    print("错误：边缘检测后没有找到任何轮廓！图片可能没有明显四边形纸张边缘")
    exit()
#轮廓按照面积从打到校排序，只取前五个最大轮廓，文档一般是画最大物体
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

#遍历前五个最大的轮廓
for c in cnts:
    peri = cv2.arcLength(c, True) #计算轮廓周长，True代表闭合图形
    # approxPolyDP轮廓近似，把曲线轮廓简化成多边形， 0.02 * peri是精度，True闭合
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 如果近似之后正好有四个顶点，说明找到了文档四边形，保存这个轮廓
    if len(approx) == 4:
        screenCnt = approx
        break

# 兜底：没有找到4角文档
if screenCnt is None:
	print("未检测到4个角的文档四边形，请更换图片，保证纸张边缘清晰！")
	exit()


# 在图片上画出找到的文档轮廓，绿色，线宽2
print("STEP2: 获取轮廓")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# step3 透视变换矫正文档
# screenCnt现在是缩放后图片上的坐标 需要乘ratio还原会原图尺寸坐标，reshape(4, 2)转化为4个点(x, y)点
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# step4 二值化：模拟扫描黑白文档效果
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) #矫正后色彩图转灰度图
# 轻微模糊，消除照片颗粒噪点
warped = cv2.GaussianBlur(warped,(3,3),0)
ref = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,3)

# 阈值二值化：大于100设置为255白色，小于100设置为0黑色：输出ref就是扫描效果
# ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
# 自适应高斯阈值，适合光线不均匀的文档照片
ref = cv2.adaptiveThreshold(
    warped,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=15,
    C=3
)

# =========新增形态学降噪，去除雪花小点=========
# 开运算：先腐蚀再膨胀，消除孤立小黑噪点
kernel = np.ones((2,2), np.uint8)
ref = cv2.morphologyEx(ref, cv2.MORPH_OPEN, kernel, iterations=1)

cv2.imwrite("scan.jpg", ref)#将扫描结果保存为scan.jpg

# 展示原图和 最终扫描效果图
print("STEP3: 变换")
cv2.imshow("Original", resize(orig, height = 650))
cv2.imshow("Scaned", resize(ref, height = 650))
cv2.waitKey(0)

























