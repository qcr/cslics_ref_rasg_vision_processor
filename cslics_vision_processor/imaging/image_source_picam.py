#!/usr/bin/env python3

import numpy, time, json, cv2
from logging import Logger
from cslics_vision_processor.imaging.arducam_focuser import ArducamFocuser
from cslics_vision_processor.imaging import ImageSource
from cslics_mqtt.comms import CameraSettings
from pathlib import Path
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import Output
from picamera2.request import CompletedRequest
from threading import RLock
from typing import Callable, List, Optional, Tuple, Union
from .picamera2_helpers import *

I2C_BUS = 10

##
# @brief class CallbackOutput(Output) - provides a callback object for picamera images
class CallbackOutput(Output):

    ##
    # @brief __init__ - initialises the picamera image callback object.
    # @param callback_on_frame_encoded : the calcv2.COLOR_YUV420p2RGBlable callback function
    def __init__(self, callback_on_frame_encoded: Callable[[bytes], None]):
        self.callback_on_frame_encoded: Callable[[bytes], None] = callback_on_frame_encoded

    ##
    # @brief outputframe - overloaded method of the class picamera2.outputs.Output for recieving image frame data.
    # @param frame : the image data in bytes
    # @param keyframe : whether the frame is a keyframe (default True)
    # @param timestamp : the timestamp of the frame
    def outputframe(self, frame: bytes, keyframe=True, timestamp=None) -> None:
        self.callback_on_frame_encoded(frame)


class PicamParameters:
    AnalogueGain: str = 'AnalogueGain'
    AutoExposureEnable: str = 'AeEnable'
    AutoExposureMode: str = 'AeExposureMode'
    AutoExposureConstraintMode: str = 'AeConstraintMode'
    AutoWhiteBalanceEnable: str = 'AwbEnable'
    AutoWhiteBalanceMode: str = 'AwbMode'
    ColourGains: str = 'ColourGains'
    ColourGainsBlue: str = 'ColourGains_Blue'
    ColourGainsRed: str = 'ColourGains_Red'
    Contrast: str = 'Contrast'
    ExposureTime: str = 'ExposureTime'
    ExposureValue: str = 'ExposureValue'
    Saturation: str = 'Saturation'
    Sharpness: str = 'Sharpness'


class PiCamConfiguration:
    InFocusVolume: str = 'InFocusVolume'

    additional_parameters: List[str] = [
        InFocusVolume
    ]

    def __init__(self, config_path: Path, logger: Logger):
        self.__controls: dict = {}

        self.__in_focus_volume: float
        
        if not config_path.exists():
            raise FileNotFoundError(f'Unable to find PiCam configuration at path: {config_path.absolute()}')

        # Get the auto white balance algorithm
        white_balance_algorithm = Picamera2.find_tuning_algo(Picamera2.load_tuning_file('imx477.json'), 'rpi.awb')
        self.__colour_temperature_curve: List[float] = white_balance_algorithm['ct_curve']
        
        with open(config_path) as file:
            self.__load(json.load(file), logger)
    
    @property
    def in_focus_volume(self) -> float:
        return self.__in_focus_volume

    def __load(self, settings: dict, logger: Logger) -> None:
        # Required parameters
        self.__in_focus_volume = settings[PiCamConfiguration.InFocusVolume]

        optional_parameter_loaders: dict[str, Callable[[any], None]] = {
            PicamParameters.AnalogueGain: self.apply_analogue_gain,
            PicamParameters.AutoExposureEnable: self.apply_auto_exposure_enable,
            PicamParameters.AutoExposureConstraintMode: self.apply_auto_exposure_constraint_mode,
            PicamParameters.AutoExposureMode: self.apply_auto_exposure_mode,
            PicamParameters.AutoWhiteBalanceEnable: self.apply_auto_white_balance_enable,
            PicamParameters.AutoWhiteBalanceMode: self.apply_auto_white_balance_mode,
            PicamParameters.ColourGainsBlue: self.apply_colour_gains_blue,
            PicamParameters.ColourGainsRed: self.apply_colour_gains_red,
            PicamParameters.Contrast: self.apply_contrast,
            PicamParameters.ExposureTime: self.apply_exposure_time,
            PicamParameters.ExposureValue: self.apply_exposure_value,
            PicamParameters.Saturation: self.apply_saturation,
            PicamParameters.Sharpness: self.apply_sharpness
        }

        for key, value in settings.items():
            if key not in optional_parameter_loaders:
                if key in PiCamConfiguration.additional_parameters:
                    continue
                
                logger.warning(f'Unrecognised configuration parameter with name "{key}"!')
                continue
            
            optional_parameter_loaders[key](value)
    
    def apply_analogue_gain(self, value: float) -> None:
        self.__controls[PicamParameters.AnalogueGain] = float(value)

    def apply_auto_exposure_enable(self, value: bool) -> None:
        self.__controls[PicamParameters.AutoExposureEnable] = bool(value)

    def apply_auto_exposure_constraint_mode(self, value: Union[controls.AeConstraintModeEnum, str]) -> None:
        mode: controls.AeConstraintModeEnum = value if isinstance(value, controls.AeConstraintModeEnum) else as_AeConstraintModeEnum(value)
        self.__controls[PicamParameters.AutoExposureConstraintMode] = mode

    def apply_auto_exposure_mode(self, value: Union[controls.AeExposureModeEnum, str]) -> None:
        mode: controls.AeExposureModeEnum = value if isinstance(value, controls.AeExposureModeEnum) else as_AeExposureModeEnum(value)
        self.__controls[PicamParameters.AutoExposureMode] = mode

    def apply_auto_white_balance_enable(self, value: bool) -> None:
        self.__controls[PicamParameters.AutoWhiteBalanceEnable] = bool(value)

    def apply_auto_white_balance_mode(self, value: Union[controls.AwbModeEnum, str]) -> None:
        mode: controls.AwbModeEnum = value if isinstance(value, controls.AwbModeEnum) else as_AwbModeEnum(value)
        self.__controls[PicamParameters.AutoWhiteBalanceMode] = mode

    def apply_colour_gains_blue(self, value: float) -> None:
        self.apply_colour_gains(blue=float(value))

    def apply_colour_gains_red(self, value: float) -> None:
        self.apply_colour_gains(red=float(value))

    def apply_colour_gains(self, *, red: Optional[float] = None, blue: Optional[float] = None):
        if red is not None and blue is not None:
            self.__controls[PicamParameters.ColourGains] = (red, blue)

            return

        if PicamParameters.ColourGains in self.__controls:
            red_existing, blue_existing = self.__controls[PicamParameters.ColourGains]
        else:
            red_existing = 1.0
            blue_existing = 1.0

        if red is None:
            red = red_existing
        
        if blue is None:
            blue = blue_existing

        self.__controls[PicamParameters.ColourGains] = (red, blue)
    
    def apply_contrast(self, value: float) -> None:
        self.__controls[PicamParameters.Contrast] = float(value)

    def apply_exposure_time(self, value: int) -> None:
        self.__controls[PicamParameters.ExposureTime] = int(value)

    def apply_exposure_value(self, value: float) -> None:
        self.__controls[PicamParameters.ExposureValue] = float(value)

    def apply_saturation(self, value: float) -> None:
        self.__controls[PicamParameters.Saturation] = float(value)

    def apply_sharpness(self, value: float) -> None:
        self.__controls[PicamParameters.Sharpness] = float(value)

    def apply_camera_settings(self, settings: CameraSettings) -> None:
        self.apply_auto_exposure_enable(settings.exposure_auto)
        self.apply_exposure_time(39 * settings.exposure)

        self.apply_auto_white_balance_enable(settings.temperature_auto)
        self.apply_auto_white_balance_mode(controls.AwbModeEnum.Auto if settings.temperature_auto else controls.AwbModeEnum.Custom)
        
        if not settings.temperature_auto:
            gain_red, gain_blue = self.__sample_colour_temperature_curve(settings.temperature / 255)
            self.apply_colour_gains(red=gain_red, blue=gain_blue)

    def __sample_colour_temperature_curve(self, normalised: float) -> Tuple[float, float]:
        stride: int = 3
        temperature_min: float = self.__colour_temperature_curve[0]
        temperature_max: float = self.__colour_temperature_curve[-stride]
        temperature_requested: float = temperature_min + (temperature_max - temperature_min) * normalised

        for i in range(0, len(self.__colour_temperature_curve) - stride, stride):
            curve_temperature: float = self.__colour_temperature_curve[i]

            if curve_temperature < temperature_requested:
                continue

            temperature_curve_next: float = self.__colour_temperature_curve[i + stride]
            lerp_t: float = (temperature_requested - curve_temperature) / (temperature_curve_next - curve_temperature)

            red_lower: float = self.__colour_temperature_curve[i + 1]
            red_upper: float = self.__colour_temperature_curve[i + 1 + stride]
            red_result: float = red_lower + (red_upper - red_lower) * lerp_t

            blue_lower: float = self.__colour_temperature_curve[i + 2]
            blue_upper: float = self.__colour_temperature_curve[i + 2 + stride]
            blue_result: float = blue_lower + (blue_upper - blue_lower) * lerp_t

            return (1.0 / red_result, 1.0 / blue_result)

        return (1.0 / self.__colour_temperature_curve[-2], 1.0 / self.__colour_temperature_curve[-1])
        
    def set_camera_controls(self, camera: Picamera2) -> None:
        camera.set_controls(self.__controls)


##
# @brief class ImageSourcePiCam(ImageSource) - implements the interface 'ImageSource' of methods for pi-camera operations.
class ImageSourcePiCam(ImageSource):

    ##
    # @brief __init__ - initialises this pi-camera operation instance.
    # @param output_length_max : the maximum number of bytes in the image
    # @param callback_on_frame_raw : the frame callback function
    # @param callback_on_frame_encoded : the frame encoding callback function
    def __init__(self, output_length_max: int, callback_on_frame_raw: Callable[[numpy.ndarray], None],
                 callback_on_frame_encoded: Callable[[bytes], None], logger: Logger, config_path: Path):
        super().__init__(output_length_max, callback_on_frame_raw, callback_on_frame_encoded, logger.getChild(ImageSourcePiCam.__name__))

        # set the camera config path
        self.configuration = PiCamConfiguration(config_path, self.logger)

        self.__control_lock = RLock()

        # define a focus object variable
        self.focuser = None

        # define the initial camera settings
        self.focus = 0

        # the camera start state
        self.is_camera_started = False

        self.camera: Picamera2 = Picamera2()
        self.configuration.set_camera_controls(self.camera)

        self.update_output_length(output_length_max)
        
        self.encoder: JpegEncoder = JpegEncoder()
        self.encoder.output = CallbackOutput(self.callback_on_frame_encoded)
        
        self.camera.encode_stream_name = 'main'
        self.camera.start_encoder(self.encoder)
        
    def update_output_length(self, output_length: int) -> None:
        super().update_output_length(output_length)

        with self.__control_lock:
            camera_running: bool = self.is_camera_started

            if camera_running:
                self.stop()

            # get the actual camera image size
            width, height = self.camera.camera_properties['PixelArraySize']

            # get ration
            camera_ratio: float = height / width
            # the image size used for raw images in ML
            output_height: int = output_length
            output_width: int = output_length

            if width > height:
                output_height = int(round(output_length * camera_ratio))
            elif height > width:
                output_width = int(round(output_length / camera_ratio))

            configuration: str = self.camera.create_still_configuration(main={'size': (width, height)}, 
                                                                        lores={'size': (output_width, output_height)})

            self.camera.configure(configuration)

            if camera_running:
                self.start()
    ##
    # @brief get_dof_volume - Computes the depth-of-field volume, given the current camera focus setting.
    # @return float : the volume in mm^3
    def get_dof_volume(self) -> float:
        return self.configuration.in_focus_volume

    ##
    # @brief start - image source start control function
    def start(self) -> None:
        with self.__control_lock:
            if self.is_camera_started:
                return
            
            self.camera.start()
            self.is_camera_started = True

            if self.focuser is None:
                self.focuser = ArducamFocuser(I2C_BUS)
            
            time.sleep(2.0)


    ##
    # @brief stop - image source stop control function
    def stop(self) -> None:
        with self.__control_lock:
            if not self.is_camera_started:
                return
            
            self.camera.stop()
            self.is_camera_started = False
    
    ##
    # @brief set_settings - adjusts the focus and exposure of the pi-camera.
    # @param settings : the list of [exposure, focus] settings for the image source
    # @pre self.is_camera_started == True
    def set_settings(self, settings: CameraSettings) -> None:
        with self.__control_lock:
            self.configuration.apply_camera_settings(settings)
            self.configuration.set_camera_controls(self.camera)

            # if the new focus is different
            if settings.focus != self.focus:
                # update the setting
                self.focus = settings.focus
                # print(settings.focus, self.focus)
                # convert byte-range to device focus range
                foc = (self.focus * 1000) // 256
                # make sure it is within range
                if 0 <= foc <= 1000:
                    # set the focus value
                    self.focuser.set(self.focuser.OPT_FOCUS, foc)

    ##
    # @brief capture - the method that starts the pi-camera, requests a frame, stops the camera, and captures the frame. 
    # @param mode : The image channel in {0: 'lores', 1: 'main'}
    # @pre self.is_camera_started == False
    def capture(self, mode: int) -> None:
        with self.__control_lock:
            # If the camera is left on, it captures continuously
            self.start()
            request: CompletedRequest = self.camera.capture_request(wait=1.0, flush=True)
            self.stop()
            # if getting the full frame
            if mode == 1:
                # send the full image
                self.callback_on_frame_raw(request.make_array('main'))
            else:
                buffer = request.make_array('lores')
                # send the low-res image
                self.callback_on_frame_raw(cv2.cvtColor(buffer, cv2.COLOR_YUV420p2BGR))
            request.release()
    
    ##
    # @brief close - the method closes the pi-camera and JPEG encoder.
    def close(self) -> None:
        with self.__control_lock:
            self.camera.close()
            self.camera.stop_encoder()
