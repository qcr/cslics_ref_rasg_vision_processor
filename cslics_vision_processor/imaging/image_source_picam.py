#!/usr/bin/env python3

import cv2, json, numpy
from logging import Logger
from cslics_mqtt.comms import CameraSettings
from cslics_vision_processor.imaging.arducam_focuser import ArducamFocuser
from cslics_vision_processor.imaging import ColourTemperatureCurve, ImageSource, MatLike
from pathlib import Path
from picamera2 import Picamera2
from threading import Lock
from typing import Callable, List, Optional, Tuple, Union
from .picamera2_helpers import *

#: The I2C bus to use for communicating with the motorised lens of the camera.
I2C_BUS = 10

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
        tuning_algo = Picamera2.find_tuning_algo(Picamera2.load_tuning_file('imx477.json'), 'rpi.awb')
        # Get the colour temperature curve from the white balance algorithm.
        self.__colour_temperature_curve = ColourTemperatureCurve(tuning_algo['ct_curve'])
        
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
            gain_red, gain_blue = self.__colour_temperature_curve.sample(settings.temperature / 255)
            self.apply_colour_gains(red=(1.0 / gain_red), blue=(1.0 / gain_blue))
        
    def set_camera_controls(self, camera: Picamera2) -> None:
        """Apply the controls maintained by this class to the provided `Picamera2` instance.
        
        Args:
            camera: The camera to apply the controls to.
        """

        camera.set_controls(self.__controls)


class ImageSourcePiCam(ImageSource):
    """An `ImageSource` implementation which utilises a Picamera2 compatible camera module."""

    def __init__(self, logger: Logger, config_path: Path):
        super().__init__(logger.getChild(ImageSourcePiCam.__name__))

        # set the camera config path
        self.__controls = PiCamControls(config_path, self.logger)

        self.__control_lock = Lock()

        self.__focus = 0
        self.__focuser = ArducamFocuser(I2C_BUS)

        self.__camera: Picamera2 = Picamera2()
        self.__initialise_camera()
        self.__controls.set_camera_controls(self.__camera)
        self.__camera.start()
        
    def __initialise_camera(self) -> None:
        with self.__control_lock:
            # get the actual camera image size
            width, height = self.__camera.camera_properties['PixelArraySize']

            # WTF: RGB888 actually yields a BGR888 image and visa versa - this can be seen in the docs in section 4.2.2.2.
            configuration: dict = self.__camera.create_still_configuration(main={'size': (width, height), 'format': 'RGB888'})

            self.__camera.configure(configuration)

    def get_dof_volume(self) -> float:
        return self.__controls.in_focus_volume
    
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

    def capture(self, timeout: Optional[float], encoded_image: bool) -> Tuple[bool, Optional[MatLike]]:
        with self.__control_lock:
            result: numpy.ndarray = self.__camera.capture_array(wait=1.0)

        if not encoded_image:
            return True, result
           
        return cv2.imencode('.jpeg', result)
    
    def close(self) -> None:
        with self.__control_lock:
            self.__camera.stop()
            self.__camera.close()
