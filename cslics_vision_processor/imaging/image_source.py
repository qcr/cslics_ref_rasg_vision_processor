#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-05-31

import numpy
from cslics_mqtt.comms import CameraSettings
from logging import Logger
from typing import Callable

class ImageSource:
    """Abstract base class for image sources."""

    def __init__(self, output_length_max: int, callback_on_frame_raw: Callable[[numpy.ndarray], None], callback_on_frame_encoded: Callable[[bytes], None], logger: Logger):
        """
        Args:
            output_length_max: The largest side length to output images in for `callback_on_frame_raw`.
            callback_on_frame_raw: The callback to invoke when a new, raw image is ready.
            callback_on_frame_encoded: The callback to invoke when a new, encoded image is ready.
            logger: The logger for implementations of `ImageSource` to use.
        """

        #: The logger for implementations of `ImageSource` to use.
        self.logger: Logger = logger

        #: The largest side length to output images in for `callback_on_frame_raw`.
        self.output_length_max: int = output_length_max

        #: The callback to invoke when a new, raw image is ready.
        self.callback_on_frame_raw: Callable[[numpy.ndarray], None] = callback_on_frame_raw

        #: The callback to invoke when a new, encoded image is ready.
        self.callback_on_frame_encoded: Callable[[bytes], None] = callback_on_frame_encoded
    
    def update_output_length(self, output_length: int) -> None:
        """Update the maximum side length of images which are output to the raw frame callback.
        
        Args:
            output_length: The new maximum side length.
        """

        self.output_length_max: int = output_length

    def start(self) -> None:
        """Start streaming images from the image source."""

        pass

    def stop(self) -> None:
        """Stop streaming images from the image source."""

        pass

    def get_dof_volume(self) -> float:
        """Get the volume which is within the depth of field of the image source.
        
        Returns:
            The volume captured by the image source in millilitres.
        """

        pass
    
    def set_settings(self, settings: CameraSettings) -> None:
        """Update settings for the camera of the image source.
        
        Args:
            settings: The decoded message containing the parameters to use.
        """

        pass

    def capture(self, mode: int) -> None:
        """Capture an image with the image source.

        This function may block until the capture is completed.

        Args:
            mode: The capture mode to use.
        """

        pass

    def close(self) -> None:
        """Close connections with cameras and shut down any running threads."""

        pass
