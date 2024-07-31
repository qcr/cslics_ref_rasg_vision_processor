#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-05-31

import numpy
from typing import Callable
from logging import Logger

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
    # @brief get_focal_parameters - gets a list of focal parameters from the image source.
    # @return list - a list of focal parameters
    def get_focus(self) -> list:
        return 
    
    ##
    # @brief set_focus - adjusts the focus of the image source.
    # @param focal_settings : the list of focal settings for the image source
    def set_focus(self, focal_settings: list) -> None:
        pass

    ##
    # @brief capture - the method that captures the image from the image source. 
    def capture(self) -> None:
        """
        Blocking
        """
        pass

    ##
    # @brief close - the method that closes the resources fo this image source.
    def close(self) -> None:
        pass