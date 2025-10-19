import cv2
import numpy as np
import os

image_height=224
image_width=224


def read_image(image_path,image_height,image_width):
  image=cv2.imread(image_path)
  image=cv2.resize(image, (image_height,image_width))
  image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  return image

def min_max_normalization (image):
    float_image = image.astype(np.float32)

    # Calculate the minimum and maximum pixel values
    min_value = np.min(float_image)
    max_value = np.max(float_image)

    # Perform Min-Max normalization
    normalized_image = (float_image - min_value) / (max_value - min_value)

    return normalized_image


def apply_histogram_normalization(image):

  b_channel, g_channel, r_channel = cv2.split(image)

  normalized_b = cv2.normalize(b_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
  normalized_g = cv2.normalize(g_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
  normalized_r = cv2.normalize(r_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


  normalized_image = cv2.merge((normalized_b, normalized_g, normalized_r))

  return normalized_image


def remove_noise(image):

  median = cv2.medianBlur(image,5)

  return median

def adaptive_gamma_correction(image):
  def apply_adaptive_gamma_correction(channel, gamma):
    corrected_channel = np.power((channel / 255.0), 1.0 / gamma)
    return cv2.normalize(corrected_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


  b_channel, g_channel, r_channel = cv2.split(image)


  gamma = 1.5
  gamma_corrected_b = apply_adaptive_gamma_correction(b_channel, gamma)
  gamma_corrected_g = apply_adaptive_gamma_correction(g_channel, gamma)
  gamma_corrected_r = apply_adaptive_gamma_correction(r_channel, gamma)


  gamma_corrected_image = cv2.merge((gamma_corrected_b, gamma_corrected_g, gamma_corrected_r))


  gamma_corrected_image=min_max_normalization(gamma_corrected_image)

  return gamma_corrected_image


def preprocess_image(img_path):

    image = read_image(img_path,image_height,image_width)

    normalized_image= apply_histogram_normalization(image)
    median= remove_noise(normalized_image)
    gamma_corrected_image=adaptive_gamma_correction(median)

    return gamma_corrected_image*255
