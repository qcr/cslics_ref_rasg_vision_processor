#!/usr/bin/env python3

# Copyright 2024 Queensland University of Technology.
#
# The programming code herein is licensed to The Australian Institute of Marine Science (AIMS)
# by The Queensland University of Technology (QUT) to use for testing and validation of the
# Coral Spawn and Larvae Imaging Camera System (CSLICS).
# 
# All liabilities and guarantees for this program code and its supporting components are as stipulated
# in the relevant agreements relating to CSLICS between AIMS and QUT and by the licenses of the
# supporting components where made by a third party. QUT accepts no liability for modifications made
# to the programming code by parties other than QUT.

# Author:   Alec Tutin
# Date:     2024-05-31

from abc import ABC, abstractmethod
from cslics_mqtt.comms import CameraSettings
from cv2.typing import MatLike
from logging import Logger
from typing import Optional

class CriticalHardwareFailureError(Exception):
    pass

class ImageSource(ABC):
    """Abstract base class for image sources."""

    def __init__(self, logger: Logger):
        """
        Args:
            output_length_max: The largest side length to output images in for `callback_on_frame_raw`.
            callback_on_frame_raw: The callback to invoke when a new, raw image is ready.
            callback_on_frame_encoded: The callback to invoke when a new, encoded image is ready.
            logger: The logger for implementations of `ImageSource` to use.

        Raises:
            `CriticalHardwareFailureError`: If critical features of the `ImageSource` were not operable during initialisation.
            `SystemError`: If the `ImageSource` was unable to be acquired or configured.
        """

        #: The logger for implementations of `ImageSource` to use.
        self.logger: Logger = logger

    @abstractmethod
    def get_dof_volume(self) -> float:
        """Get the volume which is within the depth of field of the image source.
        
        Returns:
            The volume captured by the image source in millilitres.
        """

        pass
    
    @abstractmethod
    def set_settings(self, settings: CameraSettings) -> None:
        """Update settings for the camera of the image source.
        
        Args:
            settings: The decoded message containing the parameters to use.
        """

        pass

    @abstractmethod
    def capture(self, timeout: Optional[float], encoded_image: bool) -> MatLike:
        """Capture an encoded image with the image source.

        Args:
            timeout: An optional timeout for the length of time to wait for the sensor to take the image - if `None` will wait indefiniately until an image arrives.
            encoded_image: Whether the image should be encoded instead of raw.
        
        Returns:
            If an encoded image is requested, the result will be a JPEG in RGB colour order - otherwise it will be raw in BGR colour order.

        Raises:
            `ImageCaptureFailureException`: If an image cannot be taken from the `ImageSource`.
            `ImageEncodingFailureException`: If there is an encoding/decoding error for an image taken from the `ImageSource`.
        """

        pass

    @abstractmethod
    def close(self) -> None:
        """Close connections with cameras and shut down any running threads."""

        pass
