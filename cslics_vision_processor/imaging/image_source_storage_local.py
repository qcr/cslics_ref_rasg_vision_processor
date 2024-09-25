#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-06-04

import os, cv2, numpy
from typing import Callable, List
from logging import Logger
from pathlib import Path
from cslics_vision_processor.imaging import ImageSource

class ImageSourceStorageLocal(ImageSource):
    def __init__(self, output_length_max: int, callback_on_frame_raw: Callable[[numpy.ndarray], None], callback_on_frame_encoded: Callable[[bytes], None], image_directory: str, logger: Logger):
        super().__init__(output_length_max, callback_on_frame_raw, callback_on_frame_encoded, logger.getChild(ImageSourceStorageLocal.__name__))

        image_path: Path = Path(os.path.expanduser(image_directory))
        self.images: List[Path] = []
        self.image_index: int = 0

        self.latest_image: numpy.ndarray = numpy.zeros((0, 0, 3), dtype=numpy.uint8)

        if image_path.exists():
            for ext in ['jpg', 'jpeg', 'png']:
                for c in image_path.glob(f'*.{ext}'):
                    self.images.append(c)
        
        self.images.sort()
    
    def capture(self) -> None:
        if self.image_index >= len(self.images):
            self.image_index = 0
        
        image_file: Path = self.images[self.image_index]
        self.image_index += 1

        self.logger.info(f'Loading image from: {image_file}')

        buffer_original: numpy.ndarray = numpy.fromfile(image_file, dtype=numpy.uint8)
        image_original: numpy.ndarray = cv2.imdecode(buffer_original, cv2.IMREAD_COLOR)

        (height, width, depth) = image_original.shape
        original_ratio: float = height / width

        height_target: int = self.output_length_max
        width_target: int = self.output_length_max
        
        if height < width:
            height_target = int(self.output_length_max * original_ratio)
        elif width < height:
            width_target = int(self.output_length_max / original_ratio)
        
        if self.latest_image.shape[0] != height_target or self.latest_image.shape[1] != width_target:
            self.latest_image = numpy.zeros((height_target, width_target, 3), dtype=numpy.uint8)

        cv2.resize(image_original, (width_target, height_target), self.latest_image)

        self.callback_on_frame_raw(self.latest_image)
        self.callback_on_frame_encoded(cv2.imencode('.jpg', self.latest_image)[1].tobytes())
    
    def close(self) -> None:
        pass
