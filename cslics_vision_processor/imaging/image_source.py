#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-05-31

from abc import ABC, abstractmethod
from cslics_mqtt.comms import CameraSettings
from cv2.typing import MatLike
from logging import Logger
from typing import Optional, Tuple

class ImageSource(ABC):
    """Abstract base class for image sources."""

    def __init__(self, logger: Logger):
        """
        Args:
            output_length_max: The largest side length to output images in for `callback_on_frame_raw`.
            callback_on_frame_raw: The callback to invoke when a new, raw image is ready.
            callback_on_frame_encoded: The callback to invoke when a new, encoded image is ready.
            logger: The logger for implementations of `ImageSource` to use.
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
    def capture(self, timeout: Optional[float], encoded_image: bool) -> Tuple[bool, Optional[MatLike]]:
        """Capture an encoded image with the image source.

        Args:
            timeout: An optional timeout for the length of time to wait for the sensor to take the image - if `None` will wait indefiniately until an image arrives.
            encoded_image: Whether the image should be encoded instead of raw.
        
        Returns:
            Whether the image was successfully captured and the captured image. If the image is encoded, it will be a JPEG in RGB colour order - otherwise it will be raw in BGR colour order.
        """

        pass

    @abstractmethod
    def close(self) -> None:
        """Close connections with cameras and shut down any running threads."""

        pass
