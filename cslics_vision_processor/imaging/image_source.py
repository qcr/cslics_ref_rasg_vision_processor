#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-05-31

import numpy
from typing import Callable
from logging import Logger
from cslics_common.comms import CameraSettings

##
# @brief class IImageSource - interface for image sources
class ImageSource:

    ##
    # @brief __init__ - initialises this image source instance.
    # @param output_length_max : the maximum number of bytes in the image
    # @param callback_on_frame_raw : the image callback function
    # @param callback_on_frame_encoded : the image encoding callback function
    def __init__(self, output_length_max: int, callback_on_frame_raw: Callable[[numpy.ndarray], None], callback_on_frame_encoded: Callable[[bytes], None], logger: Logger):
        self.logger: Logger = logger
        self.output_length_max: int = output_length_max
        self.callback_on_frame_raw: Callable[[numpy.ndarray], None] = callback_on_frame_raw
        self.callback_on_frame_encoded: Callable[[bytes], None] = callback_on_frame_encoded
    
    ##
    # @brief update_output_length - Update the output length
    def update_output_length(self, output_length: int) -> None:
        self.output_length_max: int = output_length

    ##
    # @brief start - image source start control function
    def start(self) -> None:
        pass

    ##
    # @brief stop - image source stop control function
    def stop(self) -> None:
        pass

    ##
    # @brief get_dof_volume - Computes the DoF volume, given the current camera focus setting.
    # @return float : the volume
    def get_dof_volume(self) -> float:
        pass
    
    ##
    # @brief set_settings - sets the camera settings of the image source.
    # @param settings : the set of camera settings for the image source
    def set_settings(self, settings: CameraSettings) -> None:
        pass

    ##
    # @brief capture - the method that captures the image from the image source.
    # @param mode :  A camera mode provision
    def capture(self, mode: int) -> None:
        """
        Blocking
        """
        pass

    ##
    # @brief close - the method that closes the resources fo this image source.
    def close(self) -> None:
        pass