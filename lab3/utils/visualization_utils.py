import numpy as np
from math import ceil, floor
from typing import Union, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.transform_utils import Transform

def get_cmap(n, name="coolwarm"):
    """
    Usage:
    cmap=get_cmap(len(data))
    plot(..., color=cmap(0))
    """
    return plt.cm.get_cmap(name, n)


class Canvas:
    def __init__(
        self,
        size=(6, 6),
        title="Trajectories",
        axis_lim=[-0.7, 0.7, -0.7, 0.7, 0.0, 1.0],
    ):
        self.fig = plt.figure(figsize=size)
        plt.ion()
        self.ax = self.fig.add_subplot(projection="3d")
        self.title = title
        self.intrinsic = [540.0, 540.0, 320.0, 240.0, 640, 480]  # 用于可视化相机的内参
        self.axis_lim = axis_lim
        self.reset()

    def update_mark(self, extrinsic, height=0.2, **kw):
        assert isinstance(extrinsic, Transform)
        # self.reset()
        self.draw_pyramid(
            extrinsic.inverse().as_matrix(),  # 注意这边取反
            intrinsic=self.intrinsic,
            height=height,
            **kw
        )
        plt.pause(0.001)

    def reset(self):
        self.ax.clear()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_xlim(self.axis_lim[0], self.axis_lim[1])  # X轴范围
        self.ax.set_ylim(self.axis_lim[2], self.axis_lim[3])  # Y轴范围
        self.ax.set_zlim(self.axis_lim[4], self.axis_lim[5])  # Z轴范围
        self.ax.set_title(self.title)

    def keep_XYZ_aspect(self):
        plt.gca().set_box_aspect((1, 1, 1))
        range = (
            max(
                self.ax.get_xlim()[1] - self.ax.get_xlim()[0],
                self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
                self.ax.get_zlim()[1] - self.ax.get_zlim()[0],
            )
            * 0.5
        )
        x_middle = (self.ax.get_xlim()[0] + self.ax.get_xlim()[1]) * 0.5
        y_middle = (self.ax.get_ylim()[0] + self.ax.get_ylim()[1]) * 0.5
        z_middle = (self.ax.get_zlim()[0] + self.ax.get_zlim()[1]) * 0.5
        self.ax.set_xlim((x_middle - range, x_middle + range))
        self.ax.set_ylim((y_middle - range, y_middle + range))
        self.ax.set_zlim((z_middle - range, z_middle + range))

    def add_camera_at_origin(self):
        # 原点画一个四棱锥表示相机
        length = 0.08
        width = 0.06
        height = 0.1
        # 四棱
        self.ax.plot([0, length], [0, width], [0, height], "black")
        self.ax.plot([0, length], [0, -width], [0, height], "black")
        self.ax.plot([0, -length], [0, width], [0, height], "black")
        self.ax.plot([0, -length], [0, -width], [0, height], "black")
        # 方形
        self.ax.plot(
            [length, length, -length, -length, length],
            [width, -width, -width, width, width],
            [height, height, height, height, height],
            "black",
        )

    def add_plane_at_origin(self, length=0.08, width=0.06):
        self.ax.plot(
            [length, length, -length, -length, length],
            [width, -width, -width, width, width],
            [0, 0, 0, 0, 0],
            "black",
        )

    def draw_axes(self, pose_t, pose_R=None, title="", axis_size=0.15, with_plane=True):
        """x red; y green; z blue
        pose_t: (3)
        pose_R: (3, 3)
        """
        if pose_R is None:
            assert (
                type(pose_t) == Transform
            ), "Input parameter should be a Transform object or a pose_t and pose_R"
            pose_R = pose_t.rotation.as_matrix()
            pose_t = pose_t.translation
        else:
            assert len(pose_t) == 3, "pose_t should be a 3D vector"
            assert pose_R.shape == (3, 3), "pose_R should be a 3x3 matrix"

        if title != "":
            self.ax.text(pose_t[0], pose_t[1], pose_t[2], title)
        self.ax.scatter(pose_t[0], pose_t[1], pose_t[2])
        # 根据 3*3 旋转矩阵画单位坐标轴
        axis_x = pose_R.dot([axis_size, 0, 0])
        axis_y = pose_R.dot([0, axis_size, 0])
        axis_z = pose_R.dot([0, 0, axis_size])
        self.ax.plot(
            [pose_t[0], pose_t[0] + axis_x[0]],
            [pose_t[1], pose_t[1] + axis_x[1]],
            [pose_t[2], pose_t[2] + axis_x[2]],
            "r-",
        )
        self.ax.plot(
            [pose_t[0], pose_t[0] + axis_y[0]],
            [pose_t[1], pose_t[1] + axis_y[1]],
            [pose_t[2], pose_t[2] + axis_y[2]],
            "g-",
        )
        self.ax.plot(
            [pose_t[0], pose_t[0] + axis_z[0]],
            [pose_t[1], pose_t[1] + axis_z[1]],
            [pose_t[2], pose_t[2] + axis_z[2]],
            "b-",
        )
        if with_plane:
            # 画个正方形
            length = axis_size / 2
            width = axis_size / 2
            corners = [
                [
                    (pose_t + pose_R.dot([length, width, 0])),
                    (pose_t + pose_R.dot([length, -width, 0])),
                    (pose_t + pose_R.dot([-length, -width, 0])),
                    (pose_t + pose_R.dot([-length, width, 0])),
                ]
            ]
            self.ax.add_collection3d(
                Poly3DCollection(
                    corners,
                    facecolors="cyan",
                    linewidths=1,
                    edgecolors="cyan",
                    alpha=0.2,
                )
            )

    def draw_pyramid(
        self,
        extrinsic,
        color: Union[str, Tuple[float, float, float]] = "r",
        alpha=0.35,
        height=0.3,
        intrinsic=None,
        fov=None,
        label=None,
    ):
        """可视化相机, 像素坐标(0,0)画黑点(即左上角), (0,0)-(W,0)画黑线(即上边)
        Args:
            extrinsic: 相机外参 (4, 4)矩阵
            height: 锥高度
            intrinsic: 相机内参 (fx, fy, cx, cy, W, H)
            fov: 宽高比 (W/(2*fx), H/(2*fy))
                [相机视角与内参的关系 - 简书](https://www.jianshu.com/p/935044175ca4)
            注: intrinsic 和 fov 二选一

        original code from: https://github.com/demul/extrinsic2pyramid

        Example:
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(projection='3d')

        >>> intrinsic = [540.0, 540.0, 320.0, 240.0, 640, 480]
        >>> extrinsic = np.array([[-0.7818,  0.6235,  0.    ,  0.0238],
                                [ 0.3117,  0.3909, -0.866 , -0.1054],
                                [-0.54  , -0.6771, -0.5   ,  0.7826],
                                [ 0.    ,  0.    ,  0.    ,  1.    ]])
        >>> draw_pyramid(ax, extrinsic, intrinsic=intrinsic, height=0.1, label='camera')
        """
        if intrinsic is not None:
            fov_w = intrinsic[4] / intrinsic[0] / 2.0
            fov_h = intrinsic[5] / intrinsic[1] / 2.0
        elif fov is not None:
            fov_w = fov[0]
            fov_h = fov[1]
        else:
            fov_w = 0.5
            fov_h = 0.5
        vertex_std = np.array(
            [
                [0, 0, 0, 1],  # 四棱锥顶点
                [height * fov_w, -height * fov_h, height, 1],
                [height * fov_w, height * fov_h, height, 1],
                [-height * fov_w, height * fov_h, height, 1],
                [-height * fov_w, -height * fov_h, height, 1],
            ]
        )

        # 从相机坐标系变换到世界坐标系
        vertex_transformed = vertex_std @ np.linalg.inv(extrinsic).T

        self.ax.scatter(
            vertex_transformed[:4, 0],
            vertex_transformed[:4, 1],
            vertex_transformed[:4, 2],
            color=color,
            s=10,
        )
        meshes = [
            [
                vertex_transformed[0, :-1],
                vertex_transformed[1, :-1],
                vertex_transformed[2, :-1],
            ],
            [
                vertex_transformed[0, :-1],
                vertex_transformed[2, :-1],
                vertex_transformed[3, :-1],
            ],
            [
                vertex_transformed[0, :-1],
                vertex_transformed[3, :-1],
                vertex_transformed[4, :-1],
            ],
            [
                vertex_transformed[0, :-1],
                vertex_transformed[4, :-1],
                vertex_transformed[1, :-1],
            ],
            [
                vertex_transformed[1, :-1],
                vertex_transformed[2, :-1],
                vertex_transformed[3, :-1],
                vertex_transformed[4, :-1],
            ],
        ]
        poly = self.ax.add_collection3d(
            Poly3DCollection(
                meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=alpha
            )
        )

        # 特殊标记
        self.ax.scatter(
            vertex_transformed[4, 0],
            vertex_transformed[4, 1],
            vertex_transformed[4, 2],
            color="black",
            s=10,
        )
        self.ax.plot3D(
            [vertex_transformed[1, 0], vertex_transformed[4, 0]],  # x 轴坐标
            [vertex_transformed[1, 1], vertex_transformed[4, 1]],  # y 轴坐标
            [vertex_transformed[1, 2], vertex_transformed[4, 2]],  # z 轴坐标
            c="black",
            linewidth=2,
        )

        if label is not None:
            self.ax.text(
                vertex_transformed[0, 0],
                vertex_transformed[0, 1],
                vertex_transformed[0, 2],
                label,
            )

    def sphere(self, r=5, origin=(0, 0, 0), color="b", alpha=0.3, resolution=100):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        x += origin[0]
        y += origin[1]
        z += origin[2]

        self.ax.plot_surface(x, y, z, color=color, alpha=alpha)

    def cylinder(
        self, r=5, origin=(0, 0, 0), height=10, color="b", alpha=0.3, resolution=100
    ):
        z = np.linspace(-height / 2, height / 2, resolution)
        theta = np.linspace(0, 2 * np.pi, resolution)
        theta, z = np.meshgrid(theta, z)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        x += origin[0]
        y += origin[1]
        z += origin[2]

        self.ax.plot_surface(x, y, z, color=color, alpha=alpha)

    def get_img(self):
        refesh_rate = 30  # 固定刷新率
        if not hasattr(self, "buf"):  # first time
            import io
            import cv2
            import time

            self.buf = io.BytesIO()
            plt.savefig(self.buf, format="png")
            self.buf.seek(0)
            self._img = np.frombuffer(self.buf.getvalue(), dtype=np.uint8)
            self._img = cv2.imdecode(self._img, cv2.IMREAD_COLOR)
            self.last_retreive_time = time.time()

        if time.time() - self.last_retreive_time < 1 / refesh_rate:
            return self._img
        else:
            plt.savefig(self.buf, format="png")
            self.buf.seek(0)
            self._img = np.frombuffer(self.buf.getvalue(), dtype=np.uint8)
            self._img = cv2.imdecode(self._img, cv2.IMREAD_COLOR)
            self.last_retreive_time = time.time()
            return self._img
