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

#: The I2C bus to use for communicating with the motorised lens of the camera.
I2C_BUS = 10

class CallbackOutput(Output):
    """The callback class which redirects encoded output from the picamera to another callback."""

    def __init__(self, callback_on_frame_encoded: Callable[[bytes], None]):
        """
        Args:
            callback_on_frame_encoded: The callback to redirect the encoded output to.
        """

        self.callback_on_frame_encoded: Callable[[bytes], None] = callback_on_frame_encoded

    def outputframe(self, frame: bytes, keyframe=True, timestamp=None) -> None:
        self.callback_on_frame_encoded(frame)


class PicamParameters:
    """A static class housing the names for configuration items for `PiCamControls`."""

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


class PiCamControls:
    """A class for maintaining a single `picamera2` configuration, built up from multiple sources."""

    #: The name of the additional parameter for the volume of the image which is in-focus for the camera.
    InFocusVolume: str = 'InFocusVolume'

    additional_parameters: List[str] = [
        InFocusVolume
    ]

    def __init__(self, config_path: Path, logger: Logger):
        """
        Args:
            config_path: The path to the configuration file.
            logger: The logger to use for parsing warnings.
        """

        self.__controls: dict = {}

        self.__in_focus_volume: float
        
        if not config_path.exists():
            raise FileNotFoundError(f'Unable to find PiCam configuration at path: {config_path.absolute()}')

        # Get the auto white balance algorithm
        white_balance_algorithm = Picamera2.find_tuning_algo(Picamera2.load_tuning_file('imx477.json'), 'rpi.awb')
        # Get the colour temperature curve from the white balance algorithm.
        self.__colour_temperature_curve: List[float] = white_balance_algorithm['ct_curve']
        
        with open(config_path) as file:
            self.__load(json.load(file), logger)
    
    @property
    def in_focus_volume(self) -> float:
        """The in-focus volume, as specified in the configuration, in millilitres."""

        return self.__in_focus_volume

    def __load(self, settings: dict, logger: Logger) -> None:
        """Apply any supported configuration items from the `settings` dictionary to the configuration.

        Args:
            settings: The dictionary to apply configuration items from.
            logger: The logger to use for warnings.
        """

        # Required parameters
        self.__in_focus_volume = settings[PiCamControls.InFocusVolume]

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
                if key in PiCamControls.additional_parameters:
                    continue
                
                logger.warning(f'Unrecognised configuration parameter with name "{key}"!')
                continue
            
            optional_parameter_loaders[key](value)
    
    def apply_analogue_gain(self, value: float) -> None:
        """Update the `PicamParameters.AnalogueGain` control.
        
        Args:
            value: The new value.
        """

        self.__controls[PicamParameters.AnalogueGain] = float(value)

    def apply_auto_exposure_enable(self, value: bool) -> None:
        """Update the `PicamParameters.AutoExposureEnable` control.
        
        Args:
            value: The new value.
        """

        self.__controls[PicamParameters.AutoExposureEnable] = bool(value)

    def apply_auto_exposure_constraint_mode(self, value: Union[controls.AeConstraintModeEnum, str]) -> None:
        """Update the `PicamParameters.AutoExposureConstraintMode` control.
        
        Args:
            value: The new value as either the enum value or its serialised name.
        """

        mode: controls.AeConstraintModeEnum = value if isinstance(value, controls.AeConstraintModeEnum) else as_AeConstraintModeEnum(value)
        self.__controls[PicamParameters.AutoExposureConstraintMode] = mode

    def apply_auto_exposure_mode(self, value: Union[controls.AeExposureModeEnum, str]) -> None:
        """Update the `PicamParameters.AutoExposureMode` control.
        
        Args:
            value: The new value as either the enum value or its serialised name.
        """

        mode: controls.AeExposureModeEnum = value if isinstance(value, controls.AeExposureModeEnum) else as_AeExposureModeEnum(value)
        self.__controls[PicamParameters.AutoExposureMode] = mode

    def apply_auto_white_balance_enable(self, value: bool) -> None:
        """Update the `PicamParameters.AutoWhiteBalanceEnable` control.
        
        Args:
            value: The new value.
        """

        self.__controls[PicamParameters.AutoWhiteBalanceEnable] = bool(value)

    def apply_auto_white_balance_mode(self, value: Union[controls.AwbModeEnum, str]) -> None:
        """Update the `PicamParameters.AutoWhiteBalanceMode` control.
        
        Args:
            value: The new value as either the enum value or its serialised name.
        """

        mode: controls.AwbModeEnum = value if isinstance(value, controls.AwbModeEnum) else as_AwbModeEnum(value)
        self.__controls[PicamParameters.AutoWhiteBalanceMode] = mode

    def apply_colour_gains_blue(self, value: float) -> None:
        """Update the `PicamParameters.ColourGainsBlue` control.
        
        Args:
            value: The new value.
        """

        self.apply_colour_gains(blue=float(value))

    def apply_colour_gains_red(self, value: float) -> None:
        """Update the `PicamParameters.ColourGainsRed` control.
        
        Args:
            value: The new value.
        """

        self.apply_colour_gains(red=float(value))

    def apply_colour_gains(self, *, red: Optional[float] = None, blue: Optional[float] = None) -> None:
        """Update the `PicamParameters.ColourGains` control.

        Args:
            red: The normalised value of the red gain. A value of `None` will leave this control as it was.
            blue: The normalised value of the blue gain. A value of `None` will leave this control as it was.
        """

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
        """Update the `PicamParameters.Contrast` control.
        
        Args:
            value: The new value.
        """

        self.__controls[PicamParameters.Contrast] = float(value)

    def apply_exposure_time(self, value: int) -> None:
        """Update the `PicamParameters.ExposureTime` control.
        
        Args:
            value: The new value.
        """

        self.__controls[PicamParameters.ExposureTime] = int(value)

    def apply_exposure_value(self, value: float) -> None:
        """Update the `PicamParameters.ExposureValue` control.
        
        Args:
            value: The new value.
        """

        self.__controls[PicamParameters.ExposureValue] = float(value)

    def apply_saturation(self, value: float) -> None:
        """Update the `PicamParameters.Saturation` control.
        
        Args:
            value: The new value.
        """

        self.__controls[PicamParameters.Saturation] = float(value)

    def apply_sharpness(self, value: float) -> None:
        """Update the `PicamParameters.Sharpness` control.
        
        Args:
            value: The new value.
        """

        self.__controls[PicamParameters.Sharpness] = float(value)

    def apply_camera_settings(self, settings: CameraSettings) -> None:
        """Update all compatible parameters from a `CameraSettings` message to the configuration.
        
        Args:
            settings: The `CameraSettings` message.
        """

        self.apply_auto_exposure_enable(settings.exposure_auto)
        self.apply_exposure_time(39 * settings.exposure)

        self.apply_auto_white_balance_enable(settings.temperature_auto)
        self.apply_auto_white_balance_mode(controls.AwbModeEnum.Auto if settings.temperature_auto else controls.AwbModeEnum.Custom)
        
        if not settings.temperature_auto:
            gain_red, gain_blue = self.__sample_colour_temperature_curve(settings.temperature / 255)
            self.apply_colour_gains(red=gain_red, blue=gain_blue)

    def __sample_colour_temperature_curve(self, normalised: float) -> Tuple[float, float]:
        """Given a normalised requested colour temperature, sample the curves to produce red and blue gains.
        
        Args:
            normalised: The normalised colour temperature request value.

        Returns:
            A tuple of the red and blue gains.
        """

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
        """Apply the controls maintained by this class to the provided `Picamera2` instance.
        
        Args:
            camera: The camera to apply the controls to.
        """

        camera.set_controls(self.__controls)


class ImageSourcePiCam(ImageSource):
    """An `ImageSource` implementation which utilises a Picamera2 compatible camera module."""

    def __init__(self, output_length_max: int, callback_on_frame_raw: Callable[[numpy.ndarray], None],
                 callback_on_frame_encoded: Callable[[bytes], None], logger: Logger, config_path: Path):
        super().__init__(output_length_max, callback_on_frame_raw, callback_on_frame_encoded, logger.getChild(ImageSourcePiCam.__name__))

        # set the camera config path
        self.__controls = PiCamControls(config_path, self.logger)

        self.__control_lock = RLock()

        # define a focus object variable
        self.__focuser = None

        # define the initial camera settings
        self.__focus = 0

        # the camera start state
        self.__is_camera_started = False

        self.__camera: Picamera2 = Picamera2()
        self.__controls.set_camera_controls(self.__camera)

        self.update_output_length(output_length_max)
        
        self.__encoder: JpegEncoder = JpegEncoder()
        self.__encoder.output = CallbackOutput(self.callback_on_frame_encoded)
        
        self.__camera.encode_stream_name = 'main'
        self.__camera.start_encoder(self.__encoder)
        
    def update_output_length(self, output_length: int) -> None:
        super().update_output_length(output_length)

        with self.__control_lock:
            camera_running: bool = self.__is_camera_started

            if camera_running:
                self.stop()

            # get the actual camera image size
            width, height = self.__camera.camera_properties['PixelArraySize']

            # get ration
            camera_ratio: float = height / width
            # the image size used for raw images in ML
            output_height: int = output_length
            output_width: int = output_length

            if width > height:
                output_height = int(round(output_length * camera_ratio))
            elif height > width:
                output_width = int(round(output_length / camera_ratio))

            configuration: dict = self.__camera.create_still_configuration(main={'size': (width, height)}, 
                                                                        lores={'size': (output_width, output_height)})

            self.__camera.configure(configuration)

            if camera_running:
                self.start()

    def get_dof_volume(self) -> float:
        return self.__controls.in_focus_volume

    def start(self) -> None:
        with self.__control_lock:
            if self.__is_camera_started:
                return
            
            self.__camera.start()
            self.__controls.set_camera_controls(self.__camera)
            self.__is_camera_started = True

            if self.__focuser is None:
                self.__focuser = ArducamFocuser(I2C_BUS)
            
            time.sleep(2.0)

    def stop(self) -> None:
        with self.__control_lock:
            if not self.__is_camera_started:
                return
            
            self.__camera.stop()
            self.__is_camera_started = False
    
    def set_settings(self, settings: CameraSettings) -> None:
        with self.__control_lock:
            self.__controls.apply_camera_settings(settings)
            self.__controls.set_camera_controls(self.__camera)

            # if the new focus is different
            if settings.focus != self.__focus:
                # update the setting
                self.__focus = settings.focus
                # print(settings.focus, self.focus)
                # convert byte-range to device focus range
                foc = (self.__focus * 1000) // 256
                # make sure it is within range
                if 0 <= foc <= 1000:
                    # set the focus value
                    self.__focuser.set(self.__focuser.OPT_FOCUS, foc)

    def capture(self, mode: int) -> None:
        with self.__control_lock:
            # If the camera is left on, it captures continuously
            self.start()
            request: CompletedRequest = self.__camera.capture_request(wait=1.0, flush=True)
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
    
    def close(self) -> None:
        with self.__control_lock:
            self.__camera.close()
            self.__camera.stop_encoder()
