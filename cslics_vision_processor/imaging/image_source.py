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
