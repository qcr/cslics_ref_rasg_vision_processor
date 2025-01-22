"""A package providing the image sources for `cslics_vision_processor`.

Author:
    Alec Tutin and Rune Rasmussen

Contact:
    a.tutin@qut.edu.au
"""

from .image_source import *

try:
    # Dependency on PiCamera2 which will only be available on RPi systems...
    from .image_source_picam import ImageSourcePiCam
except ModuleNotFoundError:
    pass

from .image_source_storage_local import ImageSourceStorageLocal
