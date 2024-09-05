#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-05-31

import numpy
import time
import json
import cv2
from typing import Callable
from logging import Logger
from cslics_vision_processor.imaging.arducam_focuser import ArducamFocuser
from cslics_vision_processor.imaging import ImageSource
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import Output
from picamera2.request import CompletedRequest

I2C_BUS = 10

##
# @brief class CallbackOutput(Output) - provides a callback object for picamera images
class CallbackOutput(Output):

    ##
    # @brief __init__ - initialises the picamera image callback object.
    # @param callback_on_frame_encoded : the callable callback function
    def __init__(self, callback_on_frame_encoded: Callable[[bytes], None]):
        self.callback_on_frame_encoded: Callable[[bytes], None] = callback_on_frame_encoded

    ##
    # @brief outputframe - overloaded method of the class picamera2.outputs.Output for recieving image frame data.
    # @param frame : the image data in bytes
    # @param keyframe : whether the frame is a keyframe (default True)
    # @param timestamp : the timestamp of the frame
    def outputframe(self, frame: bytes, keyframe=True, timestamp=None) -> None:
        self.callback_on_frame_encoded(frame)

##
# @brief class ImageSourcePiCam(ImageSource) - implements the interface 'ImageSource' of methods for pi-camera operations.
class ImageSourcePiCam(ImageSource):

    ##
    # @brief __init__ - initialises this pi-camera operation instance.
    # @param output_length_max : the maximum number of bytes in the image
    # @param callback_on_frame_raw : the frame callback function
    # @param callback_on_frame_encoded : the frame encoding callback function
    def __init__(self, output_length_max: int, callback_on_frame_raw: Callable[[numpy.ndarray], 
                None], callback_on_frame_encoded: Callable[[bytes], None], logger: Logger, config_path: str):
        super().__init__(output_length_max, callback_on_frame_raw, callback_on_frame_encoded, logger.getChild(ImageSourcePiCam.__name__))

        self.camera: Picamera2 = Picamera2()
        # set the camera config path
        self.config_path = config_path

        width, height = self.camera.camera_properties['PixelArraySize']

        print("width, height",  width, height)

        configuration: str = self.camera.create_still_configuration(main={'size': (width, height)})

        self.camera.configure(configuration)

        self.encoder: JpegEncoder = JpegEncoder()
        self.encoder.output = CallbackOutput(self.callback_on_frame_encoded)
        
        self.camera.encode_stream_name = 'main'
        self.camera.start_encoder(self.encoder)
        # define a focus object variable
        self.focuser = None
        # define an initial focus value
        self.focus = 10.0
        # the camera start state
        self.is_camera_started = False
        # if the config file is None
        if self.config_path is None:
            self.default_config()
        else:
            # load json file
            with open(self.config_path) as f:
                conf = json.load(f)
                try:
                    self.apply_conf(conf)
                except Exception as e: 
                    self.logger.error("In camera configuration file %s %s", 
                                      self.config_path, repr(e))
                f.close()
        

    ##
    # @brief start - image source start control function
    def start(self) -> None:
        if not self.is_camera_started:
            self.camera.start()
            self.is_camera_started = True
            if self.focuser is None:
                time.sleep(2)
                self.focuser = ArducamFocuser(I2C_BUS)


    ##
    # @brief stop - image source stop control function
    def stop(self) -> None:
        if self.is_camera_started:
            self.camera.stop()
            self.is_camera_started = False

    ##
    # @brief get_settings - gets the list [exposure time, focus] from the pi-camera.
    # @return list - a list of camera settings
    # @pre self.is_camera_started == True
    def get_settings(self) -> list:
        # initialise the focus value
        foc_value = -1
        # get the focus value
        foc_value = self.focuser.get(self.focuser.OPT_FOCUS)
        # return the settings
        return [self.get_exposure_time(), foc_value]
    
    ##
    # @brief set_settings - adjusts the focus and exposure of the pi-camera.
    # @param settings : the list of [exposure, focus] settings for the image source
    # @pre self.is_camera_started == True
    def set_settings(self, settings: list) -> None:
        # get the exposure time byte-range to 0,..,10000
        exp_t = 39 * settings[0]
        # convert byte-range to device focus range
        foc = (settings[1] * 1000) // 256
        print(exp_t, foc)
        # make sure it is within range
        if 0 <= foc <= 1000:
            # set the focus value
            self.focuser.set(self.focuser.OPT_FOCUS, foc)
            self.focus = foc
        # set exposure time
        self.set_exposure_time(exp_t)

    ##
    # @brief capture - the method that starts the pi-camera, requests a frame, stops the camera, and captures the frame. 
    # @pre self.is_camera_started == False
    def capture(self) -> None:
        # If the camera is left on, it captures continuously
        self.camera.start()
        request: CompletedRequest = self.camera.capture_request()
        self.camera.stop()
        # send the resized the image
        self.callback_on_frame_raw(request.make_array('main'))
        request.release()
    
    ##
    # @brief close - the method closes the pi-camera and JPEG encoder.
    def close(self) -> None:
        self.camera.close()
        self.camera.stop_encoder()

## FROM https://github.com/Coral-Imaging/coral_spawn_imager/blob/main/src/coral_spawn_imager/PiCamera2Wrapper.py

    def set_fps(self, fps: float):
        self.camera.video_configuration.controls.FrameRate = fps
        self.camera.configure("video")


    def get_colour_gains(self):
        metadata = self.camera.capture_metadata()
        return metadata['ColourGains'] # (red_gain, blue_gain)


    def set_awb(self, awb_enable: bool = True, awb_mode: str = 'Auto', red_gain = None, blue_gain = None):

        awb_mode_enum = {'Auto': 0,
                         'Tungsten': 1,
                         'Fluorescent': 2,
                         'Indoor': 3,
                         'Daylight': 4,
                         'Cloudy': 5,
                         'Custom': 6}

        # print('setting white balance')

        if (red_gain is None and blue_gain is None) or (red_gain < 0.0 and blue_gain < 0.0):
            # automatic control
            control = {'AwbEnable': awb_enable,
                       'AwbMode': awb_mode_enum[awb_mode]}
        else:
            control = {'ColourGains': (red_gain, blue_gain)}
            # setting these automatically disables AWB
        self.camera.set_controls(control)
        # there will be a delay of several frames before the controls take effect, thus we sleep for 3 seconds to allow the controls to take effect
        time.sleep(2)
    

    def get_exposure_mode(self):
        controls = self.camera.camera_controls
        aeEnable = controls['AeEnable']
        aeConstraintMode = controls['AeConstraintMode']
        return aeEnable, aeConstraintMode

    
    def set_exposure_mode(self, ae_enable= None):
                        #   ae_constraint_mode=None):

        # ae_mode_enum = {'Normal': controls.AeConstraintModeEnum.Normal,
        #                  'Highlight': controls.AeConstraintModeEnum.Highlight,
        #                  'Shadows': controls.AeConstraintModeEnum.Shadows,
        #                  'Custom': controls.AeConstraintModeEnum.Custom}
        
        ae_enable = bool(ae_enable)
        if ae_enable is not None:
            if type(ae_enable) is bool:
                self.camera.set_controls({'AeEnable': ae_enable})
            else:
                raise TypeError('ae_enable is not a valid bool')
        
        # if ae_constraint_mode is not None:
        #     self.camera.set_controls({"AeConstraintMode": ae_mode_enum[ae_constraint_mode]})

    def get_exposure_time(self):
        metadata = self.camera.capture_metadata()
        return metadata['ExposureTime'] # ms


    def set_exposure_time(self, exposure_time: int = 10000):
        # set shutter time in ms
        with self.camera.controls as controls:
            controls.ExposureTime = exposure_time


    def get_gain(self):
        metadata = self.camera.capture_metadata()
        return metadata['AnalogueGain'] # 1-4?


    def set_gain(self, gain: float = 4.0):
        # aka iso
        with self.camera.controls as controls:
            controls.AnalogueGain = gain


    def get_contrast(self):
        metadata = self.camera.capture_metadata()
        return metadata['Contrast']


    def set_contrast(self, contrast: float = 1.0):
        with self.camera.controls as controls:
            controls.Contrast = contrast


    def get_noise_reduction_mode(self):
        metadata = self.camera.capture_metadata()
        return metadata['NoiseReductionMode']


    def get_saturation(self):
        metadata = self.camera.capture_metadata()
        return metadata['Saturation']


    def set_saturation(self, saturation):
        self.camera.set_controls({'Saturation': saturation})


    def get_sharpness(self):
        metadata = self.camera.capture_metadata()
        return metadata['Sharpness']


    def set_sharpness(self, sharpness):
        self.camera.set_controls({'Sharpness': sharpness})

    def default_config(self):
        self.set_exposure_mode(True)
        self.set_gain(20.0)
        self.set_awb(True, "Auto", 2.3, 2.3)
        self.set_contrast(1.0)
        self.set_exposure_time(8000)
        time.sleep(2)

    def apply_conf(self, conf):
        self.set_exposure_mode(bool(conf["AeEnable"]))
        self.set_gain(float(conf["AnalogueGain"]))
        self.set_awb(bool(conf["AwbEnable"]), str(conf["AwbMode"]), 
                     float(conf["ColourGains_Red"]), float(conf["ColourGains_Blue"]))
        self.set_contrast(float(conf["Contrast"]))
        self.set_exposure_time(int(conf["ExposureTime"]))
        # self.set_exposure_value() # not yet implemented
        # self.set_fps() # todo - receive from conf
        # print('config frame duration')
        # self.set_frame_duration_limits(conf.frame_duration_limits_min, conf.frame_duration_limits_max)
        # print('config noise reduction')
        # self.set_noise_reduction_mode(conf.noise_reduction_mode)
        # self.set_saturation(conf.saturation)
        # self.set_sharpness(conf.sharpness)
        time.sleep(2)