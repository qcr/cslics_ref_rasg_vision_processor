#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-06-04

import cv2, numpy, os
from cslics_mqtt.comms import CameraSettings
from cslics_vision_processor.imaging import ImageSource, MatLike
from logging import Logger
from pathlib import Path
from typing import List, Optional

class ImageSourceStorageLocal(ImageSource):
    """An `ImageSource` implementation which uses an image sequence in a directory to simulate a camera."""

    def __init__(self, image_directory: str, logger: Logger):
        super().__init__(logger.getChild(ImageSourceStorageLocal.__name__))

        image_path: Path = Path(os.path.expanduser(image_directory))
        self.__images: List[Path] = []
        self.__image_index: int = 0

        if image_path.exists():
            for ext in ['jpg', 'jpeg', 'png']:
                for c in image_path.glob(f'*.{ext}'):
                    self.__images.append(c)
        
        self.__images.sort()

    def __get_next_image_path(self) -> Path:
        """Get the path to the next image to load.
        
        Returns:
            The path to the next image in the sequence.
        """

        if self.__image_index >= len(self.__images):
            self.__image_index = 0
        
        image_path: Path = self.__images[self.__image_index]
        self.__image_index += 1

        return image_path
    
    def get_dof_volume(self) -> float:
        return 3
    
    def set_settings(self, settings: CameraSettings):
        pass
    
    def capture(self, timeout: Optional[float], encoded_image: bool) -> MatLike:
        image_path: Path = self.__get_next_image_path()

        self.logger.info(f'Loading capture image from: {image_path}')

        image: numpy.ndarray = numpy.fromfile(image_path, dtype=numpy.uint8)

        if encoded_image:
            return image
        
        return cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    def close(self) -> None:
        pass
