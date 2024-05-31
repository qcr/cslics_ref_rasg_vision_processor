#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-05-31

from typing import Callable

class ImageSource:
    def __init__(self, output_width: int, on_frame: Callable[[bytes], None]):
        self.output_width: int = output_width

    def capture(self) -> None:
        pass

    def close(self) -> None:
        pass