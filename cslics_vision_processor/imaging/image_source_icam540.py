#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2025-03-10

import cv2, numpy
from CamNavi2 import CamNavi2
from cslics_mqtt.comms import CameraSettings
from cslics_vision_processor.imaging import ImageSource
from logging import Logger
from threading import Event, Lock, RLock
from typing import Callable, Optional

def to_parameter_value(value: float, *, min_: int = 0, max_: int = 100) -> float:
    if value < 0.0:
        value = 0.0
    elif value > 1.0:
        value = 1.0

    return int(value * (max_ - min_)) + min_

class ImageSourceIcam540(ImageSource):
    """An `ImageSource` implementation for the Advantech ICam-540 machine vision camera."""

    def __init__(self, output_length_max: int, callback_on_frame_raw: Callable[[numpy.ndarray], None], callback_on_frame_encoded: Callable[[bytes], None], logger: Logger):
        super().__init__(output_length_max, callback_on_frame_raw, callback_on_frame_encoded, logger.getChild(ImageSourceIcam540.__name__))

        self.__control_lock = RLock()
        self.__is_camera_started: bool = False

        self.__encoded_image: Optional[bytes] = None
        self.__image_lock = Lock()
        self.__image_received = Event()

        self.__last_settings: Optional[CameraSettings] = None

        self.__cam_navi2 = CamNavi2.CamNavi2()
        self.__camera = self.__setup_camera(self.__cam_navi2)

    def __setup_camera(self, cam_navi2: CamNavi2.CamNavi2) -> any:
        """Setup the camera and light.
        
        Returns:
            The handle for the camera/light.

        Raises:
            `SystemError`: If the camera was unable to be acquired or configured.
        """

        camera = cam_navi2.get_device_by_name('iCam500')

        if camera is None:
            raise SystemError('Unable to acquire a connection to the camera subsystem.')

        self.__sensor_width: int = self.__camera.sensor_width
        self.__sensor_height: int = self.__camera.sensor_height

        pipe_params: dict = {
            'acq_mode': 1,
            'width': self.__sensor_width,
            'height': self.__sensor_height,
            'enable_infer': 0,
            'format': 'BGRA'
        }

        config_success: bool = cam_navi2.advcam_config_pipeline(self.__camera, **pipe_params) == 'Config pipeline OK'

        if not config_success:
            raise SystemError('Unable to set up camera!')
        
        cam_navi2.advcam_open(self.__camera)

        self.__cam_navi2.advcam_register_new_image_handler(self.__camera, self.__image_handler)

        self.__camera.set_acq_frame_rate(1)
        self.__camera.set_lighting_strobe_enable(0)
        self.__camera.set_lighting_pos(3)
        self.__camera.set_lighting_gain(0)
        self.__camera.set_img_auto_exposure(0)

        return camera

    def __image_handler(self, image) -> None:
        buffer = image.get_buffer()

        with self.__image_lock:
            self.__encoded_image = buffer.extract_dup(0, buffer.get_size())

        self.__image_received.set()

    def __set_light_gain(self, value: float) -> None:
        self.__camera.set_lighting_gain(to_parameter_value(value))

    def __set_image_brightness(self, value: float) -> None:
        self.__camera.set_img_brightness(to_parameter_value(value))

    def __set_image_exposure_time(self, value: float) -> None:
        self.__camera.set_img_exposure_time(to_parameter_value(value, min_=10, max_=1000))

    def __set_image_gain(self, value: float) -> None:
        self.__camera.set_img_gain(to_parameter_value(value, max_=24))

    def __set_camera_focus(self, value: float, max_step: int = 300) -> None:
        target_position: int = to_parameter_value(value, max_=1600)

        position: int = self.__camera.focus.position()
        delta: int = target_position - position

        direction: int = 1 if delta > 0 else 0
        remaining: int = abs(delta)

        while remaining > max_step:
            self.__camera.focus.distance = max_step
            self.__camera.focus.direction = direction
            remaining -= max_step

        if remaining != 0:
            self.__camera.focus.distance = remaining
            self.__camera.focus.direction = direction

    def start(self) -> None:
        with self.__control_lock:
            if self.__is_camera_started:
                return
            
            self.__cam_navi2.advcam_play(self.__camera)
            self.__is_camera_started = True

    def stop(self) -> None:
        with self.__control_lock:
            if not self.__is_camera_started:
                return
            
            self.__cam_navi2.advcam_stop(self.__camera)
            self.__is_camera_started = False

    def capture(self, mode: int) -> None:
        with self.__control_lock:
            self.__image_received.clear()
            self.start()

            if not self.__image_received.wait(10.0):
                raise SystemError('Timed out waiting to receive an image from the camera!')
            
            self.stop()
        
        with self.__image_lock:
            self.callback_on_frame_encoded(self.__encoded_image)
            decoded_image = cv2.imdecode(self.__encoded_image, cv2.IMREAD_COLOR)
        
        self.callback_on_frame_raw(decoded_image)

    def __set_settings_all(self, settings: CameraSettings) -> None:
        self.__set_camera_focus(settings.focus / 255)
        self.__set_image_exposure_time(settings.exposure / 255)
        # TODO: Image Exposure Auto
        # TODO: Colour Temperature
        # TODO: Colour Temperature Auto
        self.__set_light_gain(settings.light_intensity / 255)

        self.__last_settings = settings

    def __update_settings(self, update: CameraSettings) -> None:
        if self.__last_settings.focus != update.focus:
            self.__set_camera_focus(update.focus / 255)

        if self.__last_settings.exposure != update.exposure:
            self.__set_image_exposure_time(update.exposure / 255)
        
        # TODO: Image Exposure Auto
        # TODO: Colour Temperature
        # TODO: Colour Temperature Auto

        if self.__last_settings.light_intensity != update.light_intensity:
            self.__set_light_gain(update.light_intensity / 255)
        
        self.__last_settings = update

    def set_settings(self, settings: CameraSettings) -> None:
        with self.__control_lock:
            if self.__last_settings is None:
                self.__set_settings_all(settings)
            else:
                self.__update_settings(settings)            

    def get_dof_volume(self) -> float:
        return 3

    def close(self) -> None:
        with self.__control_lock:
            self.__camera.set_lighting_pos(0)
            self.__cam_navi2.advcam_register_new_image_handler(self.__camera, None)
            self.__cam_navi2.advcam_close(self.__camera)
