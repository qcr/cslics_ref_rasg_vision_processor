"""A package providing the image sources for `cslics_vision_processor`.

Author:
    Alec Tutin and Rune Rasmussen

Contact:
    a.tutin@qut.edu.au
"""

from .colour_temperature_curve import ColourTemperatureCurve
from .image_source import *

__pdoc__ = {}

try:
    from .image_source_icam540 import ImageSourceIcam540
except ModuleNotFoundError:
    __pdoc__['image_source_icam540'] = False

try:
    # Dependency on PiCamera2 which will only be available on RPi systems...
    from .image_source_picam import ImageSourcePiCam
except ModuleNotFoundError:
    __pdoc__['image_source_picam'] = False
    __pdoc__['picamera2_helpers'] = False

from .image_source_storage_local import ImageSourceStorageLocal
