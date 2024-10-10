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
from cslics_common.comms import CameraSettings
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import Output
from picamera2.request import CompletedRequest
from libcamera import controls

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

        # define a focus object variable
        self.focuser = None
        # define the initial camera settings
        self.focus = 128
        self.exposure = 128 
        self.exposure_auto = False
        self.temperature = 128
        self.temperature_auto = False
        self.focal_length = 12.0
        self.pixel_size_um = 1.55
        self.working_distance_mm = 50.0

        # The near/far with focus observations
        self.focus_far_near = {0: [271.5, 276.0], 250: [277.5, 282.5], 500: [281.5, 285.5], 750: [285.0, 289.0], 1000: [285.5,289.0]}

        # set the camera config path
        self.config_path = config_path

        # the camera start state
        self.is_camera_started = False

        # get the auto white balance algorithm
        self.awb_algo = Picamera2.find_tuning_algo(Picamera2.load_tuning_file("imx477.json"), "rpi.awb")

        # get the temperature curve limits
        self.t_min, self.t_max = self.get_colour_temperature_curve_limits()
        self.logger.info(f'{self.t_min}, {self.t_max}')

        self.camera: Picamera2 = Picamera2()

        self.update_output_length(output_length_max)
        
        self.encoder: JpegEncoder = JpegEncoder()
        self.encoder.output = CallbackOutput(self.callback_on_frame_encoded)
        
        self.camera.encode_stream_name = 'main'
        self.camera.start_encoder(self.encoder)

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

    def update_output_length(self, output_length: int) -> None:
        super().update_output_length(output_length)

        self.stop()

        # get the actual camera image size
        width, height = self.camera.camera_properties['PixelArraySize']

        self.logger.info(f"width: {width}, height: {height}")

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
    
    def get_colour_temperature_curve_limits(self):
        # get the Color Temperature curve
        the_curve = self.awb_algo['ct_curve']
        # get the minimum value
        min_temp = float(the_curve[0])
        # define max value
        max_temp = min_temp
        # for each element
        for val in the_curve:
            # make sure it's a float
            val = float(val)
            # if bigger
            if max_temp < val:
                # update
                max_temp = val
        # return the limits
        return min_temp, max_temp

    
    def compute_red_blue_gains(self, temp: float):
        # get the Color Temperature curve
        the_curve = self.awb_algo['ct_curve']
        # get the curve list length
        curve_len = len(the_curve)
        # upper triple
        upper_trip = None
        lower_trip = None
        # for each 3rd element
        for i in range(0, curve_len, 3):
            # get the temp
            t = float(the_curve[i])
            # if a direct match
            if t == temp:
                return t, 1.0/float(the_curve[i+1]), 1.0/float(the_curve[i+2])
            # if larger
            elif t > temp:
                # get the upper triple
                upper_trip = (t, the_curve[i+1], the_curve[i+2])
                # get the lower triple
                lower_trip = (the_curve[i-3], the_curve[i-2], the_curve[i-1])
                # break the loop
                break
        # get interpolation rate
        int_rate = (temp - lower_trip[0]) / (upper_trip[0] - lower_trip[0])
        # interpolate the r and b values
        inv_r = lower_trip[1] + int_rate * (upper_trip[1] - lower_trip[1])
        inv_b = lower_trip[2] + int_rate * (upper_trip[2] - lower_trip[2])
        # return the gains
        return temp, 1.0/inv_r, 1.0/inv_b

    ##
    # @brief set_temperature - sets the red/blue gain based in a normalised temperature value.
    # @param temp : float in [0.0, 1.0]
    def set_temperature(self, temp: float):
        # get temperature as a lookup value
        temp_lookup = self.t_min + temp * (self.t_max - self.t_min)
        # get the gain values
        _, r_gain, b_gain = self.compute_red_blue_gains(temp_lookup)
        # print("Temperature ", r_gain, b_gain)
        # if auto mode
        if self.temperature_auto:
            # set white balance
            self.set_awb(awb_mode= 'Auto', red_gain= r_gain, blue_gain= b_gain)
        else:
            # set white balance
            self.set_awb(awb_mode='Custom', red_gain= r_gain, blue_gain= b_gain)


    def get_dof_interpolated_for_focus(self, foc: float):
        # the previous key
        prev_k = 0.0
        k_t = 0.0
        lower_vals = None
        upper_vals = None
        # for each key
        for k in self.focus_far_near.keys():
            # set to float
            k = float(k)
            # if the key is equal
            if foc == k:
                return (self.focus_far_near[k][1] - self.focus_far_near[k][0])
            # if the key is greater
            if foc < k:
                k_t = (foc - prev_k) / (k - prev_k)
                upper_vals = self.focus_far_near[k]
                lower_vals = self.focus_far_near[prev_k]
                break
            # set previous
            prev_k = k
        # compute the interpoled results
        new_far = lower_vals[0] + k_t * (upper_vals[0] - lower_vals[0])
        new_near = lower_vals[1] + k_t * (upper_vals[1] - lower_vals[1])
        # return the new far and near values
        return (new_near - new_far)


    ##
    # @brief get_dof_volume - Computes the depth-of-field volume, given the current camera focus setting.
    # @return float : the volume in mm^3
    def get_dof_volume(self) -> float:
        # get the actual camera image size
        width, height = self.camera.camera_properties['PixelArraySize']
        # get the sensor witch and height in mm
        sensor_width = width * self.pixel_size_um / 1000.0 
        sensor_height = height * self.pixel_size_um / 1000.0
        # get the depth of field
        dof = self.get_dof_interpolated_for_focus(float((self.focus * 1000) // 256))
        print("DOF mm ", dof)
        # get the height and width of the average plane
        hfov = self.working_distance_mm * sensor_height / (1.33 * self.focal_length)
        vfov = self.working_distance_mm * sensor_width / (1.33 * self.focal_length)
        # return the volume
        return (hfov * vfov * dof)

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
    def get_settings(self) -> CameraSettings:
        # return the settings
        return self.camera_settings
    
    ##
    # @brief set_settings - adjusts the focus and exposure of the pi-camera.
    # @param settings : the list of [exposure, focus] settings for the image source
    # @pre self.is_camera_started == True
    def set_settings(self, settings: CameraSettings) -> None:
        # ensure the camera has started
        self.start()
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
        # if the new exposure is different
        if settings.exposure != self.exposure:
            # update the setting
            self.exposure = settings.exposure
            # get the exposure time byte-range to 0,..,10000
            exp_t = 39 * self.exposure
            # set exposure time
            self.set_exposure_time(exp_t)
        # if the new exposure mode is different
        if settings.exposure_auto != self.exposure_auto:
            # update the setting
            self.exposure_auto = settings.exposure_auto
            # if setting auto exposure
            if self.exposure_auto:
                # set exposure mode
                self.set_exposure_mode(controls.AeConstraintModeEnum.Normal)
            else:
                # set exposure mode
                self.set_exposure_mode(None)
        if settings.temperature != self.temperature or settings.temperature_auto != self.temperature_auto:
            # update the setting
            self.temperature = settings.temperature
            # update the setting
            self.temperature_auto = settings.temperature_auto
            # make sure the temperature value is valid
            if 0 <= self.temperature <= 255:
                # get temperature as a parametric
                temp = float(self.temperature) / 255.0
                # set the temperature
                self.set_temperature(temp)


        
        # self.camera.stop()

    ##
    # @brief capture - the method that starts the pi-camera, requests a frame, stops the camera, and captures the frame. 
    # @param mode : The image channel in {0: 'lores', 1: 'main'}
    # @pre self.is_camera_started == False
    def capture(self, mode: int) -> None:
        # If the camera is left on, it captures continuously
        self.camera.start()
        request: CompletedRequest = self.camera.capture_request()
        self.camera.stop()
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
        self.focal_length = float(conf["focal_length"])
        self.pixel_size_um = float(conf["pix_size_um"])
        self.working_distance_mm = float(conf["working_distance_mm"])
        # self.set_exposure_value() # not yet implemented
        # self.set_fps() # todo - receive from conf
        # print('config frame duration')
        # self.set_frame_duration_limits(conf.frame_duration_limits_min, conf.frame_duration_limits_max)
        # print('config noise reduction')
        # self.set_noise_reduction_mode(conf.noise_reduction_mode)
        # self.set_saturation(conf.saturation)
        # self.set_sharpness(conf.sharpness)
        time.sleep(2)