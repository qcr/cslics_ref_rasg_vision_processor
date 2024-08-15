#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-05-31

import numpy
import time
from typing import Callable
from logging import Logger
from cslics_vision_processor.imaging.arducam_focuser import ArducamFocuser
from cslics_vision_processor.imaging import ImageSource
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import Output
from picamera2.request import CompletedRequest

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
    def __init__(self, output_length_max: int, callback_on_frame_raw: Callable[[numpy.ndarray], None], callback_on_frame_encoded: Callable[[bytes], None], logger: Logger):
        super().__init__(output_length_max, callback_on_frame_raw, callback_on_frame_encoded, logger.getChild(ImageSourcePiCam.__name__))

        self.camera: Picamera2 = Picamera2()

        width, height = self.camera.camera_properties['PixelArraySize']
        camera_ratio: float = height / width

        output_height: int = output_length_max
        output_width: int = output_length_max

        if width > height:
            output_height = int(round(output_length_max * camera_ratio))
        elif height > width:
            output_width = int(round(output_length_max / camera_ratio))

        configuration: str = self.camera.create_still_configuration(main={'size': (output_width, output_height)})
        self.camera.configure(configuration)

        self.encoder: JpegEncoder = JpegEncoder()
        self.encoder.output = CallbackOutput(self.callback_on_frame_encoded)
        
        self.camera.encode_stream_name = 'main'
        self.camera.start_encoder(self.encoder)
        # define a focus object variable
        self.focuser = None
        # the camera start state
        self.is_camera_started = False

    ##
    # @brief start - image source start control function
    def start(self) -> None:
        if not self.is_camera_started:
            self.camera.start()
            self.is_camera_started = True
            if self.focuser is None:
                time.sleep(2)
                self.focuser = ArducamFocuser(10)


    ##
    # @brief stop - image source stop control function
    def stop(self) -> None:
        if self.is_camera_started:
            self.camera.stop()
            self.is_camera_started = False

    ##
    # @brief get_focal_parameters - gets a list of focal parameters from the pi-camera.
    # @return list - a list of focal parameters
    # @pre self.is_camera_started == True
    def get_focus(self) -> list:
        # initialise the focus value
        foc_value = -1
        # get the focus value
        foc_value = self.focuser.get(self.focuser.OPT_FOCUS)
        # return the focus
        return [foc_value]
    
    ##
    # @brief set_focus - adjusts the focus of the pi-camera.
    # @param focal_settings : the list of focal settings for the image source
    # @pre self.is_camera_started == True
    def set_focus(self, focal_settings: list) -> None:
        # convert byte to device focus range
        foc = (focal_settings[0] * 1000) // 256
        # make sure it is within range
        if 0 <= foc <= 1000:
            # set the focus value
            self.focuser.set(self.focuser.OPT_FOCUS, foc)

    ##
    # @brief capture - the method that starts the pi-camera, requests a frame, stops the camera, and captures the frame. 
    # @pre self.is_camera_started == False
    def capture(self) -> None:
        # If the camera is left on, it captures continuously
        self.camera.start()
        request: CompletedRequest = self.camera.capture_request()
        self.camera.stop()
        self.callback_on_frame_raw(request.make_array('main'))
        request.release()
    
    ##
    # @brief close - the method closes the pi-camera and JPEG encoder.
    def close(self) -> None:
        self.camera.close()
        self.camera.stop_encoder()
