#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-05-31

from typing import Callable
import numpy

class ImageSource:
    def __init__(self, output_length_max: int, callback_on_frame_raw: Callable[[numpy.ndarray], None], callback_on_frame_encoded: Callable[[bytes], None]):
        self.output_length_max: int = output_length_max
        self.callback_on_frame_raw: Callable[[numpy.ndarray], None] = callback_on_frame_raw
        self.callback_on_frame_encoded: Callable[[bytes], None] = callback_on_frame_encoded

    def capture(self) -> None:
        pass

    def close(self) -> None:
        pass