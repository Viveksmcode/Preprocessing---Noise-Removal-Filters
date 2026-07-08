# ===============================================
# Palm Leaf Manuscript Enhancement - Preprocessing
# Noise Removal Filters Implementation
# Author: VIVEK S M 
# ===============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================================
# 1️⃣ LOAD IMAGE
# ===============================================

image_path = ".png"
  # <-- Change to your image path
original = cv2.imread(image_path)

if original is None:
    raise Exception("Image not found. Check file path.")

# Convert BGR to RGB for correct display
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# Convert to Grayscale
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# ===============================================
# 2️⃣ MEDIAN FILTER
# Best for salt-and-pepper noise
# Preserves text strokes
# ===============================================

median_filtered = cv2.medianBlur(gray, 3)
# Kernel size must be odd: 3,5,7...

# ===============================================
# 3️⃣ GAUSSIAN FILTER
# Smooths image using Gaussian distribution
# ===============================================

gaussian_filtered = cv2.GaussianBlur(gray, (5,5), 0)
# (5,5) = Kernel size
# 0 = Auto sigma

# ===============================================
# 4️⃣ BILATERAL FILTER
# Smooths while preserving edges (VERY GOOD for manuscripts)
# ===============================================

bilateral_filtered = cv2.bilateralFilter(
    gray,
    d=9,              # Diameter of pixel neighborhood
    sigmaColor=75,    # Intensity similarity
    sigmaSpace=75     # Spatial closeness
)

# ===============================================
# 5️⃣ DISPLAY RESULTS
# ===============================================

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(gray, cmap='gray')
plt.title("Original Grayscale")
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(median_filtered, cmap='gray')
plt.title("Median Filter")
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(gaussian_filtered, cmap='gray')
plt.title("Gaussian Filter")
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(bilateral_filtered, cmap='gray')
plt.title("Bilateral Filter")
plt.axis('off')

plt.tight_layout()
plt.show()

# ===============================================
# 6️⃣ SAVE OUTPUT IMAGES
# ===============================================

cv2.imwrite("output_median.jpg", median_filtered)
cv2.imwrite("output_gaussian.jpg", gaussian_filtered)
cv2.imwrite("output_bilateral.jpg", bilateral_filtered)

print("All filters applied successfully and images saved.")