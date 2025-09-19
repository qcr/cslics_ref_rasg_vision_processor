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
