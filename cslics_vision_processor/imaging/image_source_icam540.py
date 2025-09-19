#!/usr/bin/env python3

# Public licence for commercial/non-commercial use RRA (internationally)
# 
# Each Project IP Owner grants to, or must obtain for, each other Party and any member of the public a perpetual, irrevocable, worldwide, non-exclusive, royalty-free, non-transferable licence (including a right of sub-license to any person (in the case of GBRF including, but not limited to, the Department)) to Use the Project IP and Project Improvements, in the field of reef restoration and adaptation, for:
# 
#   (a) non-commercial purposes, educational and/or research purposes (including for the performance of Core Commonwealth Functions by the Department); and/or
#   (b) commercial purposes, whether in Australia or elsewhere.
# 
# Each Project IP Owner acknowledges that any licence granted to the Department for Core Commonwealth Functions will not be restricted to use in the field of reef restoration and adaptation.
# 
# Derivative works must be distributed with a copy of this licence which does not further restrict the rights of licensees.
# 
# Amendment providing additional Limitation of Liability for QUT
# 
# QUT does not warrant that:
# 
#   (a) software/code is fit for the Approved Purpose, or that it has any particular qualities or characteristics;
#   (b) the software/code is free from errors, viruses, worms, or similar defects;
#   (c) the use of the software/code by the Licensee will lead to any particular result; or
#   (d) the use of the software/code will not infringe the rights (including Intellectual Property rights) of any person.
# 
# Copyright (C) 2025 Queensland University of Technology
# 

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

import cv2, json, numpy, time
from cslics_vision_processor.imaging import CameraSettings, ColourTemperatureCurve, CriticalHardwareFailureError, ImageCaptureFailureException, ImageEncodingFailureException, ImageSource, MatLike
from logging import Logger
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Optional

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

                # WTF: Setting the direction executes the actual move of the lens; therefore, we must set the distance first and always set the direction!
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
    def __init__(self, camera, lock: Lock, config_path: Path):
        self.__is_running: bool = True
        self.__camera = camera
        self.__lock: Lock = lock

        with config_path.open() as config_file:
            config: dict = json.load(config_file)

        if 'ct_curve' not in config:
            raise Exception('Colour temperature curve missing from camera configuration!')

        self.__ct_curve = ColourTemperatureCurve(config['ct_curve'])

        self.__values: dict = {}

        self.__wake_thread = Event()
        self.__thread = Thread(target=self.__thread_handler, name=CameraParameterHandler.__name__)
        self.__thread.start()

    def __thread_handler(self) -> None:
        while self.__is_running:
            self.__wake_thread.wait()
            self.__wake_thread.clear()

            with self.__lock:
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

    def set_image_exposure_auto(self, value: bool) -> None:
        self.__values[self.__camera.set_img_auto_exposure] = 1 if value else 0
        self.__wake_thread.set()

    def set_image_gain(self, value: float) -> None:
        self.__values[self.__camera.set_img_gain] = to_parameter_value(value, max_=24)
        self.__wake_thread.set()

    def __set_image_awb_op(self, value: int) -> None:
        self.__camera.image.awb_op = value

    def set_image_temperature_auto(self, value: bool) -> None:
        self.__values[self.__set_image_awb_op] = 1 if value else 0
        self.__wake_thread.set()

    def __set_image_awb_rgb(self, value: float) -> None:
        # Valid range is 1 <= value <= 8188

        value_green: int = 1024

        min_: int = 512
        max_: int = min_ + 1024

        gain_red, gain_blue = self.__ct_curve.sample(value)

        # WTF: The library throws value errors if you send it odd numbers for these parameters!
        value_red: int = to_parameter_value(gain_red, min_=min_, max_=max_) * 2
        value_blue: int = to_parameter_value(gain_blue, min_=min_, max_=max_) * 2

        self.__camera.image.awb_red = value_red
        self.__camera.image.awb_green = value_green
        self.__camera.image.awb_blue = value_blue

    def set_colour_temperature(self, value: float) -> None:
        self.__values[self.__set_image_awb_rgb] = value
        self.__wake_thread.set()

    def close(self) -> None:
        self.__is_running = False
        self.__wake_thread.set()
        self.__thread.join()


class ImageSourceIcam540(ImageSource):
    """An `ImageSource` implementation for the Advantech ICam-540 machine vision camera."""

    def __init__(self, logger: Logger, config_path: Path):
        super().__init__(logger.getChild(ImageSourceIcam540.__name__))

        self.__last_settings: Optional[CameraSettings] = None

        self.__camera_lock = Lock()

        self.__cam_navi2 = CamNavi2.CamNavi2()
        self.__camera = self.__setup_camera(self.__cam_navi2, config_path)

        # Check if the camera is actually operable.
        try:
            self.capture(5.0, False)
        except ImageCaptureFailureException as e:
            self.close()
            raise CriticalHardwareFailureError(e)

    def __setup_camera(self, cam_navi2: CamNavi2.CamNavi2, config_path: Path) -> any:
        """Setup the camera and light.
        
        Returns:
            The handle for the camera/light.

        Raises:
            `SystemError`: If the camera was unable to be acquired or configured.
        """

        print(f'Camera list: {cam_navi2.enum_camera_list()}')
        timeout: float = time.monotonic() + 10.0

        while timeout > time.monotonic():
            try:
                camera = cam_navi2.get_device_by_name('iCam500')
                break
            except:
                # On boot the GPIO takes time to get set up - hammering it until it responds resulted in unreliable behaviour.
                time.sleep(1.0)
                camera = None

        if camera is None:
            raise SystemError('Unable to acquire a connection to the camera subsystem.')

        self.__image_height: int = camera.sensor_height
        self.__image_width: int = camera.sensor_width

        self.__crop_width: int = self.__image_height // 3 * 4
        self.__crop_width_out: int = self.__image_width - self.__crop_width

        pipe_params: dict = {
            'width': self.__image_width,
            'height': self.__image_height,
            'enable_infer': 0,
            'acq_mode': 1, # 0 == streaming, 1 == software triggered. # WTF: Streaming mode locks up - do not use it!
            'format': 'YUY2', # WTF: YUY2 seems to suffer (slightly) less from the seizing issue still present in software triggered mode...
            'pipeline_mode': 'default', # 'default' == JPEG, 'simple' == raw.
            'timestamp': 0,
            'jpg_qty': 90
        }

        config_success: bool = cam_navi2.advcam_config_pipeline(camera, **pipe_params) == 'Config pipeline OK'

        if not config_success:
            raise SystemError('Unable to set up camera!')
        
        cam_navi2.advcam_open(camera)

        self.__on_image_received = Event()
        self.__image_lock = Lock()
        self.__image: Optional[bytes] = None
        cam_navi2.advcam_register_new_image_handler(camera, self.__image_handler)

        cam_navi2.advcam_play(camera)

        camera.set_lighting_strobe_enable(1)
        camera.set_lighting_pos(3)
        camera.set_lighting_gain(0)

        self.__focus = FocusHandler(camera)
        self.__parameters = CameraParameterHandler(camera, self.__camera_lock, config_path)

        return camera

    def __image_handler(self, image) -> None:
        buffer = image.get_buffer()
        image = buffer.extract_dup(0, buffer.get_size())

        with self.__image_lock:
            self.__image = image
        
        self.__on_image_received.set()

    def capture(self, timeout: Optional[float], encoded_image: bool) -> MatLike:
        self.__on_image_received.clear()

        remaining: Optional[float] = timeout
        image_wait: float = 0.5
        success: bool = False

        while not success:
            with self.__camera_lock:
                self.__camera.software_trigger()
            
            if self.__on_image_received.wait(image_wait):
                success = True
                break

            self.logger.warning(f'Sent a software trigger without receiving a response within {image_wait} seconds!')

            if remaining is not None:
                remaining -= image_wait
                
                if remaining <= 0.0:
                    break

        if not success:
            raise ImageCaptureFailureException()
        
        with self.__image_lock:
            image_array_jpeg: numpy.ndarray = numpy.frombuffer(self.__image, dtype=numpy.uint8)

        image_array_bgr = cv2.imdecode(image_array_jpeg, cv2.IMREAD_COLOR)
        image_array_bgr = image_array_bgr[:, self.__crop_width_out : self.__crop_width + self.__crop_width_out, :]

        if encoded_image:
            success, image_encoded = cv2.imencode('.jpeg', image_array_bgr)

            if not success:
                raise ImageEncodingFailureException()

            return image_encoded

        return image_array_bgr

    def __set_settings_all(self, settings: CameraSettings) -> None:
        self.__focus.set_focus(settings.focus / 255)
        self.__parameters.set_image_exposure_time(settings.exposure / 255)
        self.__parameters.set_image_exposure_auto(settings.exposure_auto)
        self.__parameters.set_colour_temperature(settings.temperature / 255)
        self.__parameters.set_image_temperature_auto(settings.temperature_auto)
        self.__parameters.set_lighting_gain(settings.light_intensity / 255)

        self.__last_settings = settings

    def __update_settings(self, update: CameraSettings) -> None:
        if self.__last_settings.focus != update.focus:
            self.__focus.set_focus(update.focus / 255)

        if self.__last_settings.exposure != update.exposure:
            self.__parameters.set_image_exposure_time(update.exposure / 255)
        
        if self.__last_settings.exposure_auto != update.exposure_auto:
            self.__parameters.set_image_exposure_auto(update.exposure_auto)

        if self.__last_settings.temperature != update.temperature:
            self.__parameters.set_colour_temperature(update.temperature / 255)

        if self.__last_settings.temperature_auto != update.temperature_auto:
            self.__parameters.set_image_temperature_auto(update.temperature_auto)

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
        self.__focus.close()
        self.__parameters.close()
        self.__camera.set_lighting_pos(0)
        self.__cam_navi2.advcam_register_new_image_handler(self.__camera, None)
        self.__cam_navi2.advcam_close(self.__camera)
