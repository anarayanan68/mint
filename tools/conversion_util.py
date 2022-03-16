# Copyright 2021, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Toolbox for converting joint representations."""
import numpy as np
import scipy
from scipy.spatial.transform import Rotation


def rotmat2euler(angles, seq="XYZ"):
  """Converts rotation matrices to axis angles.

  Args:
    angles: np array of shape [..., 3, 3] or [..., 9].
    seq: 3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for
      intrinsic rotations, or {‘x’, ‘y’, ‘z’} for extrinsic
      rotations. Used by `scipy.spatial.transform.Rotation.as_euler`.

  Returns:
    np array of shape [..., 3].
  """
  input_shape = angles.shape
  assert input_shape[-2:] == (3, 3) or input_shape[-1] == 9, (
      f"input shape is not valid! got {input_shape}")
  if input_shape[-2:] == (3, 3):
    output_shape = input_shape[:-2] + (3,)
  else:  # input_shape[-1] == 9
    output_shape = input_shape[:-1] + (3,)

  if scipy.__version__ < "1.4.0":
    # from_dcm is renamed to from_matrix in scipy 1.4.0 and will be
    # removed in scipy 1.6.0
    r = Rotation.from_dcm(angles.reshape(-1, 3, 3))
  else:
    r = Rotation.from_matrix(angles.reshape(-1, 3, 3))
  output = r.as_euler(seq, degrees=False).reshape(output_shape)
  return output


def rotmat2aa(angles):
  """Converts rotation matrices to axis angles.

  Args:
    angles: np array of shape [..., 3, 3] or [..., 9].

  Returns:
    np array of shape [..., 3].
  """
  input_shape = angles.shape
  assert input_shape[-2:] == (3, 3) or input_shape[-1] == 9, (
      f"input shape is not valid! got {input_shape}")
  if input_shape[-2:] == (3, 3):
    output_shape = input_shape[:-2] + (3,)
  else:  # input_shape[-1] == 9
    output_shape = input_shape[:-1] + (3,)

  if scipy.__version__ < "1.4.0":
    # from_dcm is renamed to from_matrix in scipy 1.4.0 and will be
    # removed in scipy 1.6.0
    r = Rotation.from_dcm(angles.reshape(-1, 3, 3))
  else:
    r = Rotation.from_matrix(angles.reshape(-1, 3, 3))
  output = r.as_rotvec().reshape(output_shape)
  return output


def aa2rotmat(angles):
  """Converts axis angles to rotation matrices.

  Args:
    angles: np array of shape [..., 3].

  Returns:
    np array of shape [..., 9].
  """
  input_shape = angles.shape
  assert input_shape[-1] == 3, (f"input shape is not valid! got {input_shape}")
  output_shape = input_shape[:-1] + (9,)

  r = Rotation.from_rotvec(angles.reshape(-1, 3))
  if scipy.__version__ < "1.4.0":
    # as_dcm is renamed to as_matrix in scipy 1.4.0 and will be
    # removed in scipy 1.6.0
    output = r.as_dcm().reshape(output_shape)
  else:
    output = r.as_matrix().reshape(output_shape)
  return output


def get_closest_rotmat(rotmats):
  """Compute the closest valid rotmat.

  Finds the rotation matrix that is closest to the inputs in terms of the
  Frobenius norm. For each input matrix it computes the SVD as R = USV' and
  sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.

  Args:
      rotmats: np array of shape (..., 3, 3) or (..., 9).

  Returns:
      A numpy array of the same shape as the inputs.
  """
  input_shape = rotmats.shape
  assert input_shape[-2:] == (3, 3) or input_shape[-1] == 9, (
      f"input shape is not valid! got {input_shape}")
  if input_shape[-1] == 9:
    rotmats = rotmats.reshape(input_shape[:-1] + (3, 3))

  u, _, vh = np.linalg.svd(rotmats)
  r_closest = np.matmul(u, vh)

  def _eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden

  # if the determinant of UV' is -1, we must flip the sign of the last
  # column of u
  det = np.linalg.det(r_closest)  # (..., )
  iden = _eye(3, det.shape)
  iden[..., 2, 2] = np.sign(det)
  r_closest = np.matmul(np.matmul(u, iden), vh)
  return r_closest.reshape(input_shape)


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
  """
  Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
  using Gram--Schmidt orthogonalization per Section B of [1].
  Args:
      d6: 6D rotation representation, of size (*, 6)
  Returns:
      batch of rotation matrices of size (*, 3, 3)
  [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
  On the Continuity of Rotation Representations in Neural Networks.
  IEEE Conference on Computer Vision and Pattern Recognition, 2019.
  Retrieved from http://arxiv.org/abs/1812.07035
  """

  a1, a2 = d6[..., :3], d6[..., 3:]
  b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
  b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
  b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
  b3 = np.cross(b1, b2, axis=-1)
  return np.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
  """
  Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
  by dropping the last row. Note that 6D representation is not unique.
  Args:
      matrix: batch of rotation matrices of size (*, 3, 3)
  Returns:
      6D rotation representation, of size (*, 6)
  [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
  On the Continuity of Rotation Representations in Neural Networks.
  IEEE Conference on Computer Vision and Pattern Recognition, 2019.
  Retrieved from http://arxiv.org/abs/1812.07035
  """
  return np.copy(matrix[..., :2, :]).reshape(*matrix.shape[:-2], 6)