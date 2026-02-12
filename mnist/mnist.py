"""Function for dealing with the MNIST dataset."""
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'jax'
from keras.datasets import mnist
from matplotlib import pyplot as plt
from scipy import ndimage

def _convolve_image(
    img: list[list[float]],
    kernels: list[list[list[float]]]
) -> list[list[list[float]]]:
  """
  Convolves an image with a list of kernels, returning a list of convolved
  images.
  """
  convolved = []
  for kernel in kernels:
    ox = -1 if kernel.shape[0] % 2 == 0 else 0
    oy = -1 if kernel.shape[1] % 2 == 0 else 0
    convolved.append(
        ndimage.convolve(
            img,
            kernel,
            mode='constant',
            cval=0.0,
            origin=(ox, oy)))

  return convolved

def _average_pool(
    img: list[list[float]],
    pool_width: int
) -> tuple[list[float], list[float]]:
  """
  Calculates the average positive and negative values in each pool in the image
  and returns them in a pair of 1D arrays.
  """
  pool_dim = img.shape[0] // pool_width
  num_pools = pool_dim * pool_dim
  pos_pools = np.zeros(num_pools, dtype=img.dtype)
  neg_pools = np.zeros(num_pools, dtype=img.dtype)

  for row_idx, row in enumerate(img):
    pool_row = row_idx // pool_width
    if pool_row == pool_dim:
      break
    pool_row_offset = pool_row * pool_dim
    for col_idx, val in enumerate(row):
      pool_col_offset = col_idx // pool_width
      if pool_col_offset == pool_dim:
        break

      offset = pool_row_offset + pool_col_offset
      if val > 0:
        pos_pools[offset] += val
      elif val < 0:
        neg_pools[offset] -= val

  pixels_per_pool = pool_width * pool_width
  return pos_pools / pixels_per_pool, neg_pools / pixels_per_pool

def translate_values(x_value: list[list[int]]) -> list[float]:
  """
  Translates an MNIST image into an array of values suitable for creating
  spikes.
  """
  img = x_value.astype(np.float32) / 256

  kernel_33_vert = np.array(
      [[1 / 3, 1 / 3, 1 / 3], [0, 0, 0], [-1 / 3, -1 / 3, -1 / 3]],
      dtype=np.float32)
  kernel_33_horiz = np.array(
      [[1 / 3, 0, -1 / 3], [1 / 3, 0, -1 / 3], [1 / 3, 0, -1 / 3]],
      dtype=np.float32)

  convolved_images = _convolve_image(img, [kernel_33_horiz, kernel_33_vert])
  pooled_values = []
  for convolved_image in convolved_images:
    pos_pools, neg_pools = _average_pool(convolved_image, 4)
    pooled_values.append(pos_pools)
    pooled_values.append(neg_pools)

  return np.clip(np.concatenate(pooled_values) * 2, 0.0, 1.0)

def translate_values_raw(x_value: list[list[int]]) -> list[float]:
  """
  Translates an MNIST image into an array of values suitable for creating
  spikes.
  """
  return np.concatenate(_average_pool(x_value.astype(np.float32) / 256, 2))

def show(x_value: list[list[int]], title: str=None):
  """Shows an MNIST sample, for debugging."""
  if title:
    plt.title(title)
  plt.imshow(x_value, cmap='gray', vmin=0, vmax=255)
  plt.show()

def load_data() -> tuple[tuple[list, list[float]], tuple[list, list[float]]]:
  """Loads the MNIST training and testing data."""
  return mnist.load_data()
