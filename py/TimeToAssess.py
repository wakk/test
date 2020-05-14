# -*- coding: utf-8 -*-
"""
Spyder 编辑器

这是一个临时脚本文件。
"""

import cv2
start = cv2.getTickCount()
# 这里写测试代码...
end = cv2.getTickCount()
print((end - start) / cv2.getTickFrequency())



import time

start = time.clock()
# 这里写测试代码...
end = time.clock()
print(end - start)