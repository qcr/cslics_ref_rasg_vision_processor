#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-06-04

import os, cv2, numpy
from typing import Callable, Optional, List
from logging import Logger
from pathlib import Path
from cslics_vision_processor.imaging import ImageSource

class ImageSourceStorageLocal(ImageSource):
    def __init__(self, output_length_max: int, callback_on_frame_raw: Callable[[numpy.ndarray], None], callback_on_frame_encoded: Callable[[bytes], None], image_directory: str, logger: Logger):
        super().__init__(output_length_max, callback_on_frame_raw, callback_on_frame_encoded, logger.getChild(ImageSourceStorageLocal.__name__))

        image_path: Path = Path(os.path.expanduser(image_directory))
        self.images: List[Path] = []
        self.image_index: int = 0

        self.latest_path: Optional[Path] = None

        if image_path.exists():
            for ext in ['jpg', 'jpeg', 'png']:
                for c in image_path.glob(f'*.{ext}'):
                    self.images.append(c)
        
        self.images.sort()
    
    def capture(self, mode: int) -> None:
        # Updated to mimic behaviour of image_source_picam.py
        
        if self.image_index >= len(self.images):
            self.image_index = 0
        
        self.latest_path = self.images[self.image_index]
        self.image_index += 1

        self.logger.info(f'Loading image from: {self.latest_path}')

        buffer_original: numpy.ndarray = numpy.fromfile(self.latest_path, dtype=numpy.uint8)
        image_original: numpy.ndarray = cv2.imdecode(buffer_original, cv2.IMREAD_COLOR)

        if mode == 1:
            self.callback_on_frame_raw(image_original)
        else:
            image_model_size: numpy.ndarray = ImageSourceStorageLocal.resize_to_max_length(image_original, self.output_length_max)
            self.callback_on_frame_raw(cv2.cvtColor(image_model_size, cv2.COLOR_RGB2BGR))
        
        self.callback_on_frame_encoded(buffer_original.tobytes())

    def resize_to_max_length(source: numpy.ndarray, length: int) -> numpy.ndarray:
        (height, width, _) = source.shape
        original_ratio: float = height / width

        height_target: int = length
        width_target: int = length
        
        if height < width:
            height_target = int(length * original_ratio)
        elif width < height:
            width_target = int(length / original_ratio)

        return cv2.resize(source, (width_target, height_target))
    
    def get_dof_volume(self) -> float:
        return 3
