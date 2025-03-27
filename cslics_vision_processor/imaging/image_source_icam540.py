#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2025-03-10

import sys

# Erase all arguments before loading this module as it uses argparse itself! WTF?
try:
    argv = sys.argv
    sys.argv = sys.argv[:1]
    from CamNavi2 import CamNavi2
finally:
    sys.argv = argv

import copy, cv2, time
from cslics_mqtt.comms import CameraSettings
from cslics_vision_processor.imaging import ImageSource
from logging import Logger
from threading import Event, Lock, Thread
from typing import Optional, Tuple

def to_parameter_value(value: float, *, min_: int = 0, max_: int = 100) -> float:
    if value < 0.0:
        value = 0.0
    elif value > 1.0:
        value = 1.0

    return int(value * (max_ - min_)) + min_


class FocusHandler:
    def __init__(self, camera, max_step: int = 300):
        self.__max_step: int = max_step

        self.__is_running: bool = True
        self.__camera = camera

        camera.focus.pos_zero()
        self.__target_position: int = camera.focus.position()

        self.__wake_thread = Event()
        self.__thread = Thread(target=self.__thread_handler, name=FocusHandler.__name__)
        self.__thread.start()

    def __thread_handler(self) -> None:
        while self.__is_running:
            self.__wake_thread.wait()
            self.__wake_thread.clear()

            if not self.__is_running:
                return

            position: int = self.__camera.focus.position()

            while position != self.__target_position:
                delta: int = self.__target_position - position

                direction: int = 1 if delta > 0 else 0
                remaining: int = abs(delta)

                self.__camera.focus.distance = min(self.__max_step, remaining)
                self.__camera.focus.direction = direction
                
                position = self.__camera.focus.position()

    def set_focus(self, value: float) -> None:
        target: int = to_parameter_value(value, max_=1600)

        if target == self.__target_position:
            return
        
        self.__target_position = target
        self.__wake_thread.set()

    def close(self) -> None:
        self.__is_running = False
        self.__wake_thread.set()
        self.__thread.join()


class CameraParameterHandler:
    def __init__(self, camera):
        self.__is_running: bool = True
        self.__camera = camera

        self.__values: dict = {}

        self.__wake_thread = Event()
        self.__thread = Thread(target=self.__thread_handler, name=CameraParameterHandler.__name__)
        self.__thread.start()

    def __thread_handler(self) -> None:
        while self.__is_running:
            self.__wake_thread.wait()
            self.__wake_thread.clear()

            while len(self.__values) > 0:
                setter, value = self.__values.popitem()
                setter(value)

    def set_lighting_gain(self, value: float) -> None:
        self.__values[self.__camera.set_lighting_gain] = to_parameter_value(value)
        self.__wake_thread.set()

    def set_image_brightness(self, value: float) -> None:
        self.__values[self.__camera.set_img_brightness] = to_parameter_value(value, max_=255)
        self.__wake_thread.set()

    def set_image_exposure_time(self, value: float) -> None:
        self.__values[self.__camera.set_img_exposure_time] = to_parameter_value(value, min_=10, max_=1000)
        self.__wake_thread.set()

    def set_image_gain(self, value: float) -> None:
        self.__values[self.__camera.set_img_gain] = to_parameter_value(value, max_=24)
        self.__wake_thread.set()

    def close(self) -> None:
        self.__is_running = False
        self.__wake_thread.set()
        self.__thread.join()


class ImageSourceIcam540(ImageSource):
    """An `ImageSource` implementation for the Advantech ICam-540 machine vision camera."""

    def __init__(self, logger: Logger):
        super().__init__(logger.getChild(ImageSourceIcam540.__name__))

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

        print(f'Camera list: {cam_navi2.enum_camera_list()}')
        camera = cam_navi2.get_device_by_name('iCam500')

        if camera is None:
            raise SystemError('Unable to acquire a connection to the camera subsystem.')

        self.__image_width: int = camera.sensor_width
        self.__image_height: int = camera.sensor_height

        pipe_params: dict = {
            'width': self.__image_width,
            'height': self.__image_height,
            'enable_infer': 0,
            'acq_mode': 1, # 0 == streaming, 1 == software triggered.
            'format': 'BGRA',
            'timestamp': 0
        }

        config_success: bool = cam_navi2.advcam_config_pipeline(camera, **pipe_params) == 'Config pipeline OK'

        if not config_success:
            raise SystemError('Unable to set up camera!')
        
        cam_navi2.advcam_open(camera)

        self.__on_image_received = Event()
        self.__image_lock = Lock()
        self.__image: Optional[bytes] = None
        cam_navi2.advcam_register_new_image_handler(camera, self.__image_handler)

        cam_navi2.advcam_play(self.__camera)

        # Reduces chance of a lock-up of the camera system on start.
        time.sleep(5.0)

        camera.set_lighting_strobe_enable(1)
        camera.set_lighting_pos(3)
        camera.set_lighting_gain(0)

        self.__focus = FocusHandler(camera)
        self.__parameters = CameraParameterHandler(camera)

        return camera

    def __image_handler(self, image) -> None:
        buffer = image.get_buffer()
        image = buffer.extract_dup(0, buffer.get_size())

        with self.__image_lock:
            self.__image = image
        
        self.__on_image_received.set()

    def capture(self, timeout: Optional[float], encoded_image: bool) -> Tuple[bool, Optional[bytes]]:
        self.__on_image_received.clear()
        self.__camera.software_trigger()
        
        success: bool = self.__on_image_received.wait(timeout)

        if not success:
            return False, None
        
        with self.__image_lock:
            image = copy.copy(self.__image)

        if encoded_image:
            return True, image

        return True, cv2.imdecode(image, cv2.IMREAD_COLOR)

    def __set_settings_all(self, settings: CameraSettings) -> None:
        self.__focus.set_focus(settings.focus / 255)
        self.__parameters.set_image_exposure_time(settings.exposure / 255)
        # TODO: Image Exposure Auto
        # TODO: Colour Temperature
        # TODO: Colour Temperature Auto
        self.__parameters.set_lighting_gain(settings.light_intensity / 255)

        self.__last_settings = settings

    def __update_settings(self, update: CameraSettings) -> None:
        if self.__last_settings.focus != update.focus:
            self.__focus.set_focus(update.focus / 255)

        if self.__last_settings.exposure != update.exposure:
            self.__parameters.set_image_exposure_time(update.exposure / 255)
        
        # TODO: Image Exposure Auto
        # TODO: Colour Temperature
        # TODO: Colour Temperature Auto

        if self.__last_settings.light_intensity != update.light_intensity:
            self.__parameters.set_lighting_gain(update.light_intensity / 255)
        
        self.__last_settings = update

    def set_settings(self, settings: CameraSettings) -> None:
        if self.__last_settings is None:
            self.__set_settings_all(settings)
        else:
            self.__update_settings(settings)

    def get_dof_volume(self) -> float:
        return 3.0

    def close(self) -> None:
        with self.__control_lock:
            self.__camera.set_lighting_pos(0)
            self.__cam_navi2.advcam_register_new_image_handler(self.__camera, None)
            self.__cam_navi2.advcam_close(self.__camera)
