
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

IMAGE_PATH = "images/original.jpg"
RESULTS_FOLDER = "results"

os.makedirs(RESULTS_FOLDER, exist_ok=True)

print("=" * 50)
print("IMAGE PROCESSING PROJECT")
print("=" * 50)


# ==========================================
# 2. LOAD ORIGINAL IMAGE
# ==========================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Image not found! Check this path: {IMAGE_PATH}"
    )

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width = gray_image.shape

print("\nOriginal Image Loaded Successfully!")
print("Width:", width)
print("Height:", height)
print("Channels:", image.shape[2])


# ==========================================
# 3. ADD SALT AND PEPPER NOISE
# ==========================================

def add_salt_pepper_noise(image, amount=0.05):

    noisy_image = image.copy()

    total_pixels = image.shape[0] * image.shape[1]
    number_of_noise_pixels = int(total_pixels * amount)

    # Random positions for Salt Noise
    salt_y = np.random.randint(
        0, image.shape[0], number_of_noise_pixels
    )
    salt_x = np.random.randint(
        0, image.shape[1], number_of_noise_pixels
    )

    noisy_image[salt_y, salt_x] = 255

    # Random positions for Pepper Noise
    pepper_y = np.random.randint(
        0, image.shape[0], number_of_noise_pixels
    )
    pepper_x = np.random.randint(
        0, image.shape[1], number_of_noise_pixels
    )

    noisy_image[pepper_y, pepper_x] = 0

    return noisy_image


np.random.seed(42)

noisy_image = add_salt_pepper_noise(image_rgb)

print("\nSalt and Pepper Noise Added Successfully!")


# ==========================================
# 4. APPLY SMOOTHING FILTERS
# ==========================================

# Average Filter
average_filter = cv2.blur(noisy_image, (5, 5))

# Median Filter
median_filter = cv2.medianBlur(noisy_image, 5)

# Gaussian Filter
gaussian_filter = cv2.GaussianBlur(
    noisy_image,
    (5, 5),
    0
)

print("Average Filter Applied!")
print("Median Filter Applied!")
print("Gaussian Filter Applied!")


# ==========================================
# 5. FOURIER TRANSFORM
# ==========================================

# Apply 2D Fourier Transform
f_transform = np.fft.fft2(gray_image)

# Shift low frequencies to the center
f_shift = np.fft.fftshift(f_transform)

# Calculate magnitude spectrum
magnitude_spectrum = 20 * np.log(
    np.abs(f_shift) + 1
)

# Normalize Fourier image for saving
fourier_display = cv2.normalize(
    magnitude_spectrum,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

print("Fourier Transform Applied Successfully!")


# ==========================================
# 6. IMAGE SEGMENTATION - OTSU
# ==========================================

threshold_value, segmented_image = cv2.threshold(
    gray_image,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print("Otsu Threshold Value:", threshold_value)
print("Image Segmentation Completed!")


# ==========================================
# 7. SAVE INDIVIDUAL RESULTS
# ==========================================

# Convert RGB results to BGR before saving with OpenCV
cv2.imwrite(
    os.path.join(RESULTS_FOLDER, "original.png"),
    cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    os.path.join(RESULTS_FOLDER, "noisy_image.png"),
    cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    os.path.join(RESULTS_FOLDER, "average_filter.png"),
    cv2.cvtColor(average_filter, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    os.path.join(RESULTS_FOLDER, "median_filter.png"),
    cv2.cvtColor(median_filter, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    os.path.join(RESULTS_FOLDER, "gaussian_filter.png"),
    cv2.cvtColor(gaussian_filter, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    os.path.join(RESULTS_FOLDER, "fourier_transform.png"),
    fourier_display
)

cv2.imwrite(
    os.path.join(RESULTS_FOLDER, "segmentation.png"),
    segmented_image
)

print("\nAll Individual Results Saved!")


# ==========================================
# 8. DISPLAY ALL RESULTS IN ONE SUBPLOT
# ==========================================

plt.figure(figsize=(16, 14))

images = [
    (image_rgb, "1. Original Image", None),
    (noisy_image, "2. Salt and Pepper Noise", None),
    (average_filter, "3. Average Filter", None),
    (median_filter, "4. Median Filter", None),
    (gaussian_filter, "5. Gaussian Filter", None),
    (magnitude_spectrum, "6. Fourier Transform", "gray"),
    (segmented_image, "7. Otsu Segmentation", "gray")
]

for index, (img, title, color_map) in enumerate(images):

    plt.subplot(3, 3, index + 1)

    if color_map:
        plt.imshow(img, cmap=color_map)
    else:
        plt.imshow(img)

    plt.title(title, fontsize=11, fontweight="bold")
    plt.axis("off")


plt.suptitle(
    "IMAGE PROCESSING PROJECT",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save final subplot
plt.savefig(
    os.path.join(RESULTS_FOLDER, "final_results.png"),
    dpi=300,
    bbox_inches="tight"
)

print("Final Subplot Saved Successfully!")

plt.show()


# ==========================================
# 9. PROJECT SUMMARY
# ==========================================

print("\n" + "=" * 50)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 50)

print("1. Original Image")
print("2. Salt and Pepper Noise")
print("3. Average Filter")
print("4. Median Filter")
print("5. Gaussian Filter")
print("6. Fourier Transform")
print("7. Image Segmentation")

print("\nAll results are saved in the 'results' folder.")
print("Final file: results/final_results.png")

print("=" * 50)
