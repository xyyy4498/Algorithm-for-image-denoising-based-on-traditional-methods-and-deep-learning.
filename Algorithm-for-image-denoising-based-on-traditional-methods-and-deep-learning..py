import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import load_model
import numpy as np
from skimage import io, img_as_float

# 加载预训练模型
model = load_model('DnCNN_model.h5')

# 读取测试图像
image = img_as_float(io.imread('test_image.png'))

# 添加噪声
noisy_image = image + 0.1 * np.random.randn(*image.shape)

# 去噪
denoised_image = model.predict(noisy_image[np.newaxis, ..., np.newaxis])

# 显示结果
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title('Noisy Image')
plt.imshow(noisy_image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Denoised Image')
plt.imshow(denoised_image[0, ..., 0], cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.show()