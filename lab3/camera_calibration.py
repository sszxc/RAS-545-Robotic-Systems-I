# camera interface; AprilTag detection; plot 3D coordinate axis

import os
import cv2
import time
import logging
import threading
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from collections import deque, Counter
from pupil_apriltags import Detector
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

from utils.transform_utils import Transform
from utils.visualization_utils import Canvas

class Camera(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_img(self, *args, **kwargs):
        """Get one RGB image from camera"""
        pass

    @abstractmethod
    def __del__(self, *args, **kwargs):
        pass

    def limit_resolution(self, image, max_size=None) -> np.ndarray:
        # Scale the image, longer side -> max_size
        height, width = image.shape[:2]
        if max_size is None:
            assert hasattr(
                self, "visualization_resize"
            ), "give a max_size or set a visualization_resize parameter for this camera"
            max_size = self.visualization_resize
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
        return image

    def save_image(self, img=None, path=None):
        """default save to log_imgs/[time].jpg, or give a path like **/**.jpg"""
        if img is None:
            img = self.get_img()
        if path is None:
            _date = "log_imgs/" + time.strftime("%m_%d", time.localtime())
            dir_path = os.path.join(os.getcwd(), _date)
            os.makedirs(dir_path, exist_ok=True)
            _time = time.strftime("%H_%M_%S", time.localtime())
            full_path = os.path.join(dir_path, _time) + ".jpg"
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            full_path = path

        cv2.imwrite(full_path, img)
        self.log.info(f"Image saved as {full_path}.")
        return full_path

    def _load_mtx_config(
        self, config_path="/Users/Henrik/atc_silicon/calibration_output/05_06_17_31"
    ):
        """load camera intrinsic & distortion from txt file"""
        self.intrinsic = np.loadtxt(os.path.join(config_path, "mtx.txt"))
        self.distortion = np.loadtxt(os.path.join(config_path, "dist.txt"))

    def undistort(self, img):
        """
        intrinsic: cameraMatrix (3x3)
        distortion: Input vector of distortion coefficients (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements
        ↑ from OpenCV
        """
        assert hasattr(self, "intrinsic") and hasattr(
            self, "distortion"
        ), "Camera intrinsic & distortion not loaded!"
        if not hasattr(self, "mapx"):
            h, w = img.shape[:2]
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.intrinsic, self.distortion, None, self.intrinsic, (w, h), 5
            )

        dst = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)
        return dst

    def get_intrinsic_list(self):
        """Get camera [fx, fy, cx, cy]"""
        assert hasattr(self, "intrinsic"), "Camera intrinsic not loaded!"
        return [
            self.intrinsic[0, 0],
            self.intrinsic[1, 1],
            self.intrinsic[0, 2],
            self.intrinsic[1, 2],
        ]

    def is_focused(self, value_change_threshold=15.0, pixel_num_threshold=0.02):
        """Compare two frames and determine whether the picture is still changing
        value_change_threshold ↓, pixel_num_threshold ↓, sensitivity ↑
        """
        assert hasattr(
            self, "_check_focus_once"
        ), "Camera focus check method not implemented!"
        first_check = self._check_focus_once(
            value_change_threshold, pixel_num_threshold
        )
        time.sleep(0.3)
        second_check = self._check_focus_once(
            value_change_threshold, pixel_num_threshold
        )
        time.sleep(0.3)
        third_check = self._check_focus_once(
            value_change_threshold, pixel_num_threshold
        )
        return first_check and second_check and third_check


class FPSCounter:
    frame_times = deque(maxlen=60)

    @classmethod
    def cal(self, current_frame_time):
        if len(self.frame_times) == 0:  # first frame
            self.frame_times.append(current_frame_time)
            return 0
        t = (current_frame_time - self.frame_times[0]) / len(self.frame_times)
        self.frame_times.append(current_frame_time)
        return 1 / t


class CameraBufferCleanerThread(threading.Thread):
    """Define the thread that will continuously pull frames from the camera, which helps to calculate FPS"""

    def __init__(self, camera, log, name="camera-buffer-cleaner-thread"):
        self.camera = camera
        self.log = log
        self.last_frame = None
        self.stop_event = threading.Event()
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.setDaemon(True)  # 主程序退出会自动结束
        self.start()

    def run(self):
        while not self.stop_event.is_set():
            ret, self.last_frame = self.camera.read()
            if not ret:
                break  # Exit if the camera read fails
            self.fps = FPSCounter.cal(time.time())

    def stop(self):
        """Stop the thread"""
        self.log.info("Try to stop camera thread.")
        self.stop_event.set()
        self.join()  # Wait for the thread to finish
        self.log.info("Camera thread stopped.")


class RGBCamera(Camera):
    def __init__(
        self,
        source="192.168.111.111",
        intrinsic_path=None,
        target_resolution=(1920, 1080, 30),
        visualization_resize=640,
        log: logging.Logger = logging.getLogger(),
        **kwargs,
    ) -> None:
        """
        source: IP (UR Roboticq) or USB camera index (UVC Cam)
        intrinsic_path: intrinsic & distortion
        target_resolution: only for USB Camera
        """
        self.log = log
        self.log.info("Start camera initialization.")
        self.target_resolution = target_resolution
        self.source = source
        self.visualization_resize = visualization_resize
        if intrinsic_path is not None:
            self._load_mtx_config(intrinsic_path)
        if self._get_frame() is not None:  # get 1 frame for test
            self.log.info("Camera initialization succeed!")
        else:
            raise RuntimeError("Camera initialization failed!")
        super().__init__(**kwargs)

    def get_img(
        self, undistort=False, with_info_overlay=False, limit_resolution=0, zoom_in=1.0
    ):
        img = deepcopy(self._get_frame())
        if undistort:
            img = self.undistort(img)
        if zoom_in != 1.0:  # zoom in, preserves only the center of the image
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            zoom_h = int(h / zoom_in)
            zoom_w = int(w / zoom_in)
            img = img[
                center[1] - zoom_h // 2 : center[1] + zoom_h // 2,
                center[0] - zoom_w // 2 : center[0] + zoom_w // 2,
            ]
        if limit_resolution != 0:
            img = self.limit_resolution(img, max_size=limit_resolution)
        if with_info_overlay:  # Add time & FPS watermark
            if hasattr(self, "cap"):
                fps = (
                    self.cam_cleaner.fps
                )  # USB camera, use the FPS from the cleaner thread
            else:
                fps = FPSCounter.cal(time.time())  # wrist camera
            # 根据图片长边计算字体大小
            h, w = img.shape[:2]
            font_scale = max(h, w) / 1000
            cv2.putText(
                img,
                f"Time: {time.strftime('%m-%d %H:%M:%S', time.localtime())} FPS: {fps:.2f}",
                (10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=(0, 0, 0),
                thickness=5,
            )
            cv2.putText(
                img,
                f"Time: {time.strftime('%m-%d %H:%M:%S', time.localtime())} FPS: {fps:.2f}",
                (10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=(255, 255, 255),
                thickness=2,
            )
        return img

    def _get_frame(self):
        if isinstance(self.source, int):  # USB camera
            if not hasattr(self, "cap"):
                self._connect_to_camera()
            frame = self.cam_cleaner.last_frame
            if frame is None:
                try:  # try to reconnect
                    self.log.warning("Camera error! Try to reconnect...")
                    self.cap.release()
                    self._connect_to_camera()
                    frame = self.cam_cleaner.last_frame
                    self.log.info("Camera reconnected.")
                except:
                    raise RuntimeError("Camera error!")
            return frame
        else:
            raise ValueError

    def _connect_to_camera(self):
        while True:
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(3, self.target_resolution[0])
            self.cap.set(4, self.target_resolution[1])
            self.cap.set(5, self.target_resolution[2])
            resolution_fps_real = (
                int(self.cap.get(3)),
                int(self.cap.get(4)),
                int(self.cap.get(5)),
            )
            self.log.info(f"Try to set camera at {self.target_resolution}.")
            self.log.info(f"Actually running at {resolution_fps_real}.")
            # if resolution config not match, raise error
            if resolution_fps_real == self.target_resolution:
                break
            # if resolution_fps_real != self.target_resolution:
            #     raise RuntimeError('Camera resolution setting failed!')
            self.log.warning(f"Camera set failed. Going to try again. Current resolution: {resolution_fps_real}")
            del self.cap
        self.cam_cleaner = CameraBufferCleanerThread(
            self.cap, self.log
        )  # Start the cleaning thread
        while self.cam_cleaner.last_frame is None:
            time.sleep(0.01)

    def get_raw_stream(self):
        if hasattr(self, "cam_cleaner"):
            return self.cam_cleaner
        else:
            raise RuntimeError("USB Camera not connected!")

    def __del__(self):
        if hasattr(self, "cap"):
            self.cam_cleaner.stop()
            self.cap.release()


class April:
    def __init__(self, camera_params, quad_decimate=1.0, debug=0):
        self.at_detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=quad_decimate,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=debug,
        )
        self.camera_params = camera_params

    def detect(self, frame, tag_size):
        if len(frame.shape) == 3:  # RGB to Gray
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.at_detector.detect(frame, estimate_tag_pose=True,
                                       camera_params=self.camera_params, tag_size=tag_size)

    def visualize(self, frame, tags, print_result=False):
        if tags:
            font_scale = frame.shape[0] / 500
            radius = int(frame.shape[0] / 50)
            thickness = int(frame.shape[0] / 100)
            for tag in tags:
                cv2.circle(frame, tuple(tag.corners[0].astype(int)), radius, (255, 0, 0), thickness)  # left-top
                cv2.circle(frame, tuple(tag.corners[1].astype(int)), radius, (255, 0, 0), thickness)  # right-top
                cv2.circle(frame, tuple(tag.corners[2].astype(int)), radius, (255, 0, 0), thickness)  # right-bottom
                cv2.circle(frame, tuple(tag.corners[3].astype(int)), radius, (255, 0, 0), thickness)  # left-bottom
                cv2.putText(frame, str(tag.tag_id), tuple(tag.center.astype(int) + [radius, -radius]),  # slight translation, no relation with radius
                            cv2.FONT_ITALIC, font_scale, (0, 0, 255), thickness)
                # Draw coordinate axes: x red; y green; z blue
                axis_length = 0.05
                axis = np.array([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).T
                axis = np.dot(tag.pose_R, axis) + tag.pose_t
                axis = np.dot(np.array([[self.camera_params[0], 0, self.camera_params[2]],
                                        [0, self.camera_params[1], self.camera_params[3]],
                                        [0, 0, 1]]), axis)
                axis = axis[:2] / axis[2]
                cv2.line(frame, tuple(axis[:, 0].astype(int)), tuple(axis[:, 1].astype(int)), (0, 0, 255), thickness)
                cv2.line(frame, tuple(axis[:, 0].astype(int)), tuple(axis[:, 2].astype(int)), (0, 255, 0), thickness)
                cv2.line(frame, tuple(axis[:, 0].astype(int)), tuple(axis[:, 3].astype(int)), (255, 0, 0), thickness)
            if print_result:
                print(tags)
        return frame

    def _find_item_twice(self, lst):
        counter = Counter(lst)
        return [item for item, count in counter.items() if count == 2]

    def filter_dual_tag(self, original_result, better_y_using_relpos=False,
                        ignore_distance_check=False, ignore_rotation_check=False):
        '''better_y_using_relpos: when dual_tag is attached along the y-axis, the relative position between the tags is used to optimize the output y-axis direction
        '''
        # Handling dual_tag, filtering tags that appear twice (discard if more or less than twice)
        result_ids = [r.tag_id for r in original_result]
        valid_twice_ids = self._find_item_twice(result_ids)
        # Fusion of each pair of localization results
        new_result = []
        for id in valid_twice_ids:
            index = [i for i, x in enumerate(result_ids) if x == id]
            assert len(index) == 2
            r0 = original_result[index[0]]
            r1 = original_result[index[1]]
            tags_translation = np.linalg.norm(r0.pose_t - r1.pose_t)
            tags_rotation = np.degrees(np.linalg.norm((R.from_matrix(r0.pose_R).inv() * R.from_matrix(r1.pose_R)).as_rotvec()))
            # Abnormal angle detection
            if not ignore_distance_check and (tags_translation > 0.12 or tags_translation < 0.105):
                warnings.warn(f"Warning! Unexpected dual tag distance: {tags_translation:.4f}, box {id}")
            if not ignore_rotation_check and tags_rotation > 2.5:
                warnings.warn(f"Warning! Unexpected dual tag rotation: {tags_rotation:.4f}, box {id}")
            # cprint(f'dis {tags_translation:.7f}, angle {tags_rotation:.7f}', 'red')  # for debug
            
            # Average the two results
            if better_y_using_relpos:
                # use r0 r1 position to calculate y-axis direction (use tag direction to find positive)
                pose_R = R.concatenate([R.from_matrix(r0.pose_R), R.from_matrix(r1.pose_R)]).mean().as_matrix()
                y_axis = pose_R[:, 1]
                relpos = r0.pose_t.T[0] - r1.pose_t.T[0]
                if np.dot(relpos, y_axis) < 0:
                    relpos = -relpos
                y_axis = relpos / np.linalg.norm(relpos)
                # y × x -> z
                z_axis = np.cross(pose_R[:, 0], y_axis)
                # z × y -> x
                x_axis = -np.cross(z_axis, y_axis)
                # construct new pose_R
                r0.pose_R = np.column_stack([x_axis, y_axis, z_axis])
            else:  # or simply average the two tag results
                r0.pose_R = R.concatenate([R.from_matrix(r0.pose_R), R.from_matrix(r1.pose_R)]).mean().as_matrix()
            r0.pose_t.T[0] = np.mean([r0.pose_t.T[0], r1.pose_t.T[0]], axis=0)
            new_result.append(r0)
        return new_result


def get_img_files(path):
    file_list = []
    entries = os.listdir(path)

    for entry in entries:
        if entry.lower().endswith(".jpg") or entry.lower().endswith(
            ".png"
        ):  # Tranverse all paths and filter out .jpg files
            full_path = os.path.join(path, entry)
            file_list.append(full_path)
    return file_list


class Calibrator:
    def __init__(self, *args, **kwargs):
        self.frame_count = 0
        os.makedirs(self.output_path, exist_ok=True)

    def add_img(self, *args, **kwargs):
        raise NotImplementedError

    def calibrate(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, camera_node, robot_node=None):
        print(
            "Press 's' to add image, and press 'q' to start calibration.\nAfter each image is added you will see a prompt on the command line indicating whether the detection was successful or not."
        )
        data_count = 0
        while True:
            cv2.imshow(
                "Visualization",
                camera_node.get_img(with_info_overlay=True, limit_resolution=640),
            )
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord("s") or key == ord("S"):
                # while not camera_node.is_focused():
                    # print("wait for focus...")
                time.sleep(0.2)
                self.add_img(camera_node.get_img())  # , robot_node.get_pose().get_array())
                data_count += 1
                print(f"added 1 data, {data_count} in total")
        self.calibrate()
        print("Calibration done!")


class Intrinsic_Calibrator(Calibrator):
    def __init__(self, board_name="Chessboard_11_7_0.02"):
        """
        height: height of the chessboard (count the number of cross-grid points)
        width: width of the chessboard
        size: square size for each square of the chessboard (m)
        """
        height, width, self.size = map(lambda x: float(x), board_name.split("_")[1:])
        self.height = int(height)
        self.width = int(width)
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )  # Used in the cornerSubPix, mainly considers the iteration step and the number of iterations, i.e. 0.001 per change, up to 30 iterations.

        corner_pts = np.zeros(
            (self.height * self.width, 3), dtype=np.float32
        )  # Initialize the list of corner points (index of corner points)
        corner_pts[:, :2] = np.mgrid[0 : self.width, 0 : self.height].T.reshape(-1, 2)
        self.corner_pts_template = corner_pts * self.size
        self.corner_pts_list = list()
        self.img_pts_list = list()
        self.output_path = f"calibration_output/intrinsic_{time.strftime('%m_%d_%H_%M', time.localtime())}/"
        super().__init__()

    def add_img(self, img, *args):
        if not hasattr(self, "img_h"):
            self.img_h, self.img_w = img.shape[:2]
        else:
            assert (
                self.img_h == img.shape[0] and self.img_w == img.shape[1]
            )  # make sure all the resolutions match
        ret, corners, img_corners = self._find_corners(img)
        if ret:
            if not hasattr(self, "demo_img"):  # used for undistortion demo
                self.demo_img = img
            print("chessboard detection succeeded.")
            cv2.imwrite(
                os.path.join(self.output_path, "{:04d}.png".format(self.frame_count)),
                img,
            )
            img = cv2.drawChessboardCorners(
                img.copy(), (self.width, self.height), img_corners, ret
            )  # Draw rainbow lines
            cv2.imwrite(
                os.path.join(
                    self.output_path, "{:04d}_display.png".format(self.frame_count)
                ),
                img,
            )
            self.corner_pts_list.append(self.corner_pts_template)
            self.img_pts_list.append(img_corners)
            self.frame_count += 1
            return True
        else:
            print("chessboard detection failed, please try other viewpoints ... ")
            # continue
            return None

    def _find_corners(self, img):  # detect the corners of the chessboard
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, (self.width, self.height), None
        )  # ret, return is 0/1
        if ret:
            # corners = np.squeeze(corners, axis=1)
            img_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), self.criteria
            )  # perform sub-pixel level of refinement
            return (
                ret,
                np.squeeze(corners, axis=1),
                np.squeeze(img_corners, axis=1).astype(np.float32),
            )
        else:
            return ret, corners, None

    def calibrate(self, dry_run=False, select_ratio=1.0):
        num_frame = len(self.img_pts_list)
        num_select = int(select_ratio * num_frame)
        if num_select != num_frame:
            random_indices = np.random.choice(num_frame, num_select, replace=False)
            corner_pts_list = np.array(self.corner_pts_list)[random_indices]
            img_pts_list = np.array(self.img_pts_list)[random_indices]
        else:
            corner_pts_list = np.array(self.corner_pts_list)
            img_pts_list = np.array(self.img_pts_list)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            corner_pts_list, img_pts_list, (self.img_w, self.img_h), None, None
        )
        # ret is also reprojection error

        if ret:
            # new_mtx_alpha0, roi_alpha0 = cv2.getOptimalNewCameraMatrix(mtx, dist, (self.img_w, self.img_h), 0, (self.img_w, self.img_h))
            # new_mtx_alpha1, roi_alpha1 = cv2.getOptimalNewCameraMatrix(mtx, dist, (self.img_w, self.img_h), 1, (self.img_w, self.img_h))
            print(f"cameraMatrix:\n{mtx}")
            print(f"distCoeffs:\n{dist}")
            # print(f'OptimalNewCameraMatrix_alpha0:\n{new_mtx_alpha0}')
            # print(f'OptimalNewCameraMatrix_alpha1:\n{new_mtx_alpha1}')

            mtx_reprojection_error = self._cal_reprojection_error(
                corner_pts_list, img_pts_list, rvecs, tvecs, mtx, dist
            )
            print(f"mtx_reprojection_error (pixel): {mtx_reprojection_error}")
            # newmtx0_reprojection_error = self._cal_reprojection_error(self.corner_pts_list, self.img_pts_list, rvecs, tvecs, new_mtx_alpha0, dist)
            # print(f'newmtx0_reprojection_error: {newmtx0_reprojection_error}')
            # newmtx1_reprojection_error = self._cal_reprojection_error(self.corner_pts_list, self.img_pts_list, rvecs, tvecs, new_mtx_alpha1, dist)
            # print(f'newmtx1_reprojection_error: {newmtx1_reprojection_error}')

            if not dry_run:
                undistort_result = cv2.undistort(self.demo_img, mtx, dist, None)
                # x, y, w, h = roi_alpha0
                # undistort_result_alpha0 = cv2.undistort(self.demo_img, mtx, dist, None, new_mtx_alpha0)[y:y+h, x:x+w]
                # x, y, w, h = roi_alpha1
                # undistort_result_alpha1 = cv2.undistort(self.demo_img, mtx, dist, None, new_mtx_alpha1)[y:y+h, x:x+w]
                cv2.imwrite(
                    os.path.join(self.output_path, "undistort_result.jpg"),
                    undistort_result,
                )
                # cv2.imwrite(os.path.join(self.output_path, 'undistort_result_alpha0.jpg'), undistort_result_alpha0)
                # cv2.imwrite(os.path.join(self.output_path, 'undistort_result_alpha1.jpg'), undistort_result_alpha1)

                np.savetxt(os.path.join(self.output_path, "mtx.txt"), mtx)
                np.savetxt(os.path.join(self.output_path, "dist.txt"), dist)
                # np.savetxt(os.path.join(self.output_path, 'OptimalNewCameraMatrix_alpha0.txt'), new_mtx_alpha0)
                # np.savetxt(os.path.join(self.output_path, 'OptimalNewCameraMatrix_alpha1.txt'), new_mtx_alpha1)
            result_data = {
                "num_imgs": num_select,
                "reprojection_error": mtx_reprojection_error,
            }
            print("Calibration finished.")
            print("====\n")
            return result_data
        else:
            print("Fail to calibrate the camera, please try again.")
            raise RuntimeError

    def _cal_reprojection_error(
        self, corner_pts_list, img_pts_list, rvecs, tvecs, mtx, dist
    ):
        mean_error = 0
        for i in range(len(corner_pts_list)):
            reproject_points, _ = cv2.projectPoints(
                corner_pts_list[i], rvecs[i], tvecs[i], mtx, dist
            )
            error = cv2.norm(
                img_pts_list[i], reproject_points[:, 0, :], cv2.NORM_L2
            ) / len(reproject_points)
            mean_error += error
        return mean_error / len(corner_pts_list)


def pixel2base(pixel_coord, intrinsic, depth, base2camera=Transform.identity()):
    # 像素坐标系转到 base 坐标系
    # pixel_coord: [u, v] 像素坐标
    # intrinsic: 相机内参矩阵 3x3
    # depth: 该点的深度值(米)
    # base2camera: 基坐标系到相机坐标系的变换矩阵
    
    pixel_homo = np.array([pixel_coord[0], pixel_coord[1], 1])  # 构建齐次像素坐标 [u,v,1]
    camera_norm = np.linalg.inv(intrinsic) @ pixel_homo  # 像素坐标转到相机坐标系下的归一化坐标
    camera_3d = camera_norm * depth  # 乘以深度得到相机坐标系下的3D坐标
    camera_3d_homo = np.append(camera_3d, 1)  # 转为齐次坐标 [x,y,z,1]
    base_3d = base2camera.as_matrix() @ camera_3d_homo
    base_3d = base_3d[:3]  # 去除齐次项,得到基坐标系下的3D坐标 [x,y,z]
    return base_3d


if __name__ == "__main__":
    # ==== calibration intrinsic ====
    camera_node = RGBCamera(source=0)
    calibrator = Intrinsic_Calibrator(board_name="Chessboard_9_7_0.5786782609")
    calibrator.run(camera_node)
    exit()

    # ==== AprilTag ====
    camera_node = RGBCamera(
        source=1,
        intrinsic_path="calibration_output/intrinsic_03_20_17_05",
    )
    april = April(camera_node.get_intrinsic_list())
    canvas = Canvas(title="Tag Trajectories")
    tag_size = 0.064
    try:
        while True:
            img = camera_node.get_img(with_info_overlay=True)  # undistort=True
            # cv2.imshow("Live Image", img)
            result = april.detect(img, tag_size)
            cv2.imshow(
                "Live Image",
                camera_node.limit_resolution(april.visualize(img, result)),
            )
            if result:
                canvas.reset()
                canvas.add_camera_at_origin()
                for tag in result:
                    canvas.draw_axes(
                        tag.pose_t.squeeze(),
                        tag.pose_R,
                        str(tag.tag_id) + "\n" + str(tag.pose_t.transpose().round(3)),
                    )
                plt.pause(0.001)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord("s") or key == ord("S"):
                time.sleep(0.1)
                camera_node.save_image()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()


#     file_list = get_img_files(
#         "/Users/Henrik/atc_silicon/calibration_output/05_08_15_45"
#     )
#     for img_file in file_list:
#         img = cv2.imread(img_file)
#         calibrator.add_img(img, np.eye(4))

#     calibrator.calibrate()
