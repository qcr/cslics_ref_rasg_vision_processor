#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-06-04

from typing import Callable, List
import os
from pathlib import Path
import cv2
import numpy
from cslics_vision_processor.imaging import ImageSource

class ImageSourceStorageLocal(ImageSource):
    def __init__(self, output_width: int, callback_on_frame_raw: Callable[[numpy.ndarray], None], callback_on_frame_encoded: Callable[[bytes], None], image_directory: str):
        super().__init__(output_width, callback_on_frame_raw, callback_on_frame_encoded)

        image_path: Path = Path(os.path.expanduser(image_directory))
        self.images: List[Path] = []
        self.image_index: int = 0

        self.latest_image: numpy.ndarray = numpy.zeros((output_width, output_width, 3), dtype=numpy.uint8)

        if image_path.exists():
            for ext in ['jpg', 'jpeg', 'png']:
                for c in image_path.glob(f'*.{ext}'):
                    self.images.append(c)
        
        self.images.sort()
    
    def capture(self) -> None:
        if self.image_index >= len(self.images):
            return
        
        image_file: Path = self.images[self.image_index]
        self.image_index += 1

        print(f'Loading image from: {image_file}')

        buffer_original: numpy.ndarray = numpy.fromfile(image_file, dtype=numpy.uint8)
        image_original: numpy.ndarray = cv2.imdecode(buffer_original, cv2.IMREAD_COLOR)

        (height, width, depth) = image_original.shape
        
        if height < width:
            margin: int = int((width - height) * 0.5)
            image_cropped = image_original[0:height, margin:(height + margin)]
        elif width < height:
            margin: int = int((height - width) * 0.5)
            image_cropped = image_original[margin:(width + margin), 0:width]
        else:
            image_cropped = image_original

        cv2.resize(image_cropped, (self.output_width, self.output_width), self.latest_image)

        self.callback_on_frame_raw(self.latest_image)

        self.callback_on_frame_encoded(cv2.imencode('.jpg', self.latest_image)[1].tobytes())
    
    def close(self) -> None:
        pass
