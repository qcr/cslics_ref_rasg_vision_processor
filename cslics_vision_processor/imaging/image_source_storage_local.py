#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-06-04

import cv2, time, numpy, os
from cslics_vision_processor.imaging import ImageSource
from logging import Logger
from pathlib import Path
from threading import Thread
from typing import Callable, Optional, List

class ImageSourceStorageLocal(ImageSource):
    """An `ImageSource` implementation which uses an image sequence in a directory to simulate a camera."""

    def __init__(self, output_length_max: int, callback_on_frame_raw: Callable[[numpy.ndarray], None], callback_on_frame_encoded: Callable[[bytes], None], image_directory: str, logger: Logger):
        super().__init__(output_length_max, callback_on_frame_raw, callback_on_frame_encoded, logger.getChild(ImageSourceStorageLocal.__name__))

        self.__is_running: bool = True
        self.__is_streaming: bool = False

        image_path: Path = Path(os.path.expanduser(image_directory))
        self.__images: List[Path] = []
        self.__image_index: int = 0

        if image_path.exists():
            for ext in ['jpg', 'jpeg', 'png']:
                for c in image_path.glob(f'*.{ext}'):
                    self.__images.append(c)
        
        self.__images.sort()

        self.__stream_thread: Optional[Thread] = None

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

    def __stream_loop(self) -> None:
        """Stream images from the source directory at 10Hz until `self.__is_streaming` or `self.__is_running` is `False`."""

        frame_delay: float = 1.0 / 10.0

        self.logger.info('Starting stream...')

        while self.__is_running and self.__is_streaming:
            loop_start: float = time.time()
            
            image_path: Path = self.__get_next_image_path()

            with image_path.open('rb') as file:
                data: bytes = file.read()

            self.callback_on_frame_encoded(data)

            required_sleep: float = frame_delay - (time.time() - loop_start)
            
            if required_sleep <= 0.0:
                time.sleep(0.01)
            else:
                time.sleep(required_sleep)

        self.logger.info('Stream stopped...')

    def start(self) -> None:
        if self.__stream_thread is not None:
            return
        
        self.__is_streaming = True

        self.__stream_thread = Thread(target=self.__stream_loop, name=f'{ImageSourceStorageLocal.__name__}.__stream_loop')
        self.__stream_thread.start()

    def stop(self) -> None:
        self.__is_streaming = False

        if self.__stream_thread is None:
            return

        self.__stream_thread.join()
        self.__stream_thread = None
    
    def capture(self, mode: int) -> None:
        image_path: Path = self.__get_next_image_path()

        self.logger.info(f'Loading capture image from: {image_path}')

        image_original: numpy.ndarray = cv2.imdecode(numpy.fromfile(image_path, dtype=numpy.uint8), cv2.IMREAD_COLOR)

        if mode == 1:
            self.callback_on_frame_raw(image_original)
        else:
            image_model_size: numpy.ndarray = ImageSourceStorageLocal.resize_to_max_length(image_original, self.output_length_max)
            self.callback_on_frame_raw(cv2.cvtColor(image_model_size, cv2.COLOR_RGB2BGR))

    def resize_to_max_length(source: numpy.ndarray, length: int) -> numpy.ndarray:
        """Resize an image to have a specified maximum side length, taking into account the image's aspect ratio.

        Args:
            source: The image to resize.
            length: The maximum side length.

        Returns:
            The image, resized to retain the aspect ratio with one side length of `length` and the other the same length or smaller.
        """
        
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
    
    def close(self) -> None:
        self.stop()
