import numpy as np
import scipy.spatial.transform
from scipy.spatial.transform import Rotation as R
from typing import Union, Optional, Literal


class Rotation(scipy.spatial.transform.Rotation):
    @classmethod
    def identity(cls):
        return cls.from_quat([0.0, 0.0, 0.0, 1.0])


class Transform(object):
    """Rigid spatial transform between coordinate systems in 3D space.

    Attributes:
        rotation (scipy.spatial.transform.Rotation)
        translation (np.ndarray)
    """

    def __init__(self, rotation, translation):
        assert isinstance(rotation, scipy.spatial.transform.Rotation)
        assert isinstance(translation, (np.ndarray, list))

        self.rotation = rotation
        self.translation = np.asarray(translation, np.double)
        # check dimension
        assert self.translation.shape == (3,)

    def as_matrix(self):
        """Represent as a 4x4 matrix."""
        return np.vstack(
            (np.c_[self.rotation.as_matrix(), self.translation], [0.0, 0.0, 0.0, 1.0])
        )

    def to_dict(self):
        """Serialize Transform object into a dictionary."""
        return {
            "rotation": self.rotation.as_quat().tolist(),
            "translation": self.translation.tolist(),
        }

    def to_list(self):
        return np.r_[self.rotation.as_quat(), self.translation]

    def to_ur_pose(self):
        '''matches the ur_robot.getl() format
        '''
        return np.r_[self.translation, self.rotation.as_rotvec()]
    
    @classmethod
    def from_ur_pose(cls, pose):
        '''matches the ur_robot.getl() format
        '''
        return cls(R.from_rotvec(pose[3:]), pose[:3])

    def to_string(self):
        return f"translation {self.translation}, rotation (euler xyz degree) {self.rotation.as_euler('xyz', degrees=True)}"

    def __mul__(self, other):
        """Compose this transform with another."""
        rotation = self.rotation * other.rotation
        translation = self.rotation.apply(other.translation) + self.translation
        return self.__class__(rotation, translation)

    def transform_point(self, point):
        return self.rotation.apply(point) + self.translation

    def transform_vector(self, vector):
        return self.rotation.apply(vector)

    def inverse(self):
        """Compute the inverse of this transform."""
        rotation = self.rotation.inv()
        translation = -rotation.apply(self.translation)
        return self.__class__(rotation, translation)

    @classmethod
    def from_translation(cls, translation):
        rotation = Rotation.identity()
        return cls(rotation, translation)

    @classmethod
    def from_matrix(cls, m):
        """Initialize from a 4x4 matrix."""
        rotation = Rotation.from_matrix(m[:3, :3])
        translation = m[:3, 3]
        return cls(rotation, translation)

    @classmethod
    def from_dict(cls, dictionary):
        rotation = Rotation.from_quat(dictionary["rotation"])
        translation = np.asarray(dictionary["translation"])
        return cls(rotation, translation)

    @classmethod
    def from_list(cls, list):
        rotation = Rotation.from_quat(list[:4])
        translation = list[4:]
        return cls(rotation, translation)

    @classmethod
    def identity(cls):
        """Initialize with the identity transformation."""
        rotation = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
        translation = np.array([0.0, 0.0, 0.0])
        return cls(rotation, translation)


def average_transforms(transforms: list[Transform]) -> Transform:
    """Compute the average of a list of Transform objects.
    Example:
    t1 = Transform.from_matrix(np.array([
        [0.999, -0.001, 0.018, -0.005],
        [0.001, 0.999, 0.018, -0.006],
        [-0.018, -0.018, 0.999, -0.175],
        [0.000, 0.000, 0.000, 1.000]
    ]))
    t2 = Transform.from_matrix(np.array([
        [0.999, -0.002, 0.015, -0.004],
        [0.002, 1.000, -0.001, -0.001],
        [-0.015, 0.001, 0.999, -0.173],
        [0.000, 0.000, 0.000, 1.000]
    ]))
    average_t = average_transforms([t1, t2])
    """
    a_translation = np.mean([t.translation for t in transforms], axis=0)
    a_rotation = Rotation.concatenate([t.rotation for t in transforms]).mean()
    return Transform(a_rotation, a_translation)


def is_upside_down(transform: Transform):
    '''If the y-axis is pointing downward, return False, otherwise return True.
    '''
    # axis_y = transform.rotation.as_matrix().dot([0, 1, 0])
    # result = axis_y[2] < 0
    return transform.rotation.as_matrix()[2, 1] > 0

def distance_between(t1: Transform, t2: Transform):
    '''Compute the translation/rotation distance between two Transform objects.
    * rotation in radians
    * input are commutative
    '''
    translation_diff = np.linalg.norm(t1.translation - t2.translation)
    rotation_diff = 2 * np.arccos((t1.rotation.inv() * t2.rotation).as_quat()[-1])
    if rotation_diff > np.pi:
        rotation_diff = 2 * np.pi - rotation_diff
    return translation_diff, rotation_diff

def align_with_base_axes(pose: Transform, relax_axis: Optional[Literal['x', 'y', 'z']] = None):
    '''Project Euler angles to multiples of 90 degrees, only tested for very small angles
    '''
    if relax_axis is None:
        euler_angles = pose.rotation.as_euler('xyz', degrees=True)
        projected_euler_angles = np.round(euler_angles / 90) * 90  
        projected_rotation = R.from_euler('xyz', projected_euler_angles, degrees=True)
    elif relax_axis =='x':
        euler_angles = pose.rotation.as_euler('xyz', degrees=True)
        projected_euler_angles = np.round(euler_angles / 90) * 90
        projected_euler_angles[0] = euler_angles[0]
        projected_rotation = R.from_euler('xyz', projected_euler_angles, degrees=True)
    elif relax_axis =='y':
        euler_angles = pose.rotation.as_euler('yzx', degrees=True)
        projected_euler_angles = np.round(euler_angles / 90) * 90
        projected_euler_angles[0] = euler_angles[0]
        projected_rotation = R.from_euler('yzx', projected_euler_angles, degrees=True)
    elif relax_axis =='z':
        euler_angles = pose.rotation.as_euler('zxy', degrees=True)
        projected_euler_angles = np.round(euler_angles / 90) * 90
        projected_euler_angles[0] = euler_angles[0]
        projected_rotation = R.from_euler('zxy', projected_euler_angles, degrees=True)

    just_t = Transform(projected_rotation, pose.translation)
    return just_t


def generate_interpolation(l1, l2, num_points):
    '''interpolate between two poses
    l1, l2: ur_robot.getl() format
    '''
    robotl_list = []
    for i in range(num_points + 2):  # +2 是为了包含左右端点
        t = i / (num_points + 1)  # 插值系数从0到1
        x = l1[0] + t * (l2[0] - l1[0])
        y = l1[1] + t * (l2[1] - l1[1]) 
        z = l1[2] + t * (l2[2] - l1[2])
        # 其他姿态角度保持不变
        robotl = [x, y, z, l1[3], l1[4], l1[5]]
        robotl_list.append(robotl)
    return robotl_list
