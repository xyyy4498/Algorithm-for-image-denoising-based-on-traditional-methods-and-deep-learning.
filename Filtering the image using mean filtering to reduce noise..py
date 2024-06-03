import cv2  
  
# 读取图像  
img_path = 'noise2.PNG'  
img = cv2.imread(img_path)  
  
if img is None:  
    print("Error: Could not read the image.")  
    exit()  
  
# 应用均值滤波  
blur = cv2.blur(img, (5, 5))  # 假设滤波器窗口大小为5x5  
  
# 显示结果  
cv2.imshow('Mean Filter Result', blur)  
cv2.waitKey(0)  
  
# 保存处理后的图像  
output_path = 'denoise2.PNG'  
cv2.imwrite(output_path, blur)  
  
# 关闭所有窗口  
cv2.destroyAllWindows()