#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-05-31

from typing import Callable
from cslics_vision_processor.imaging.image_source import ImageSource
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import Output

class CallbackOutput(Output):
    def __init__(self, callback_on_frame: Callable[[bytes], None]):
        self.callback_on_frame: Callable = callback_on_frame

    def outputframe(self, frame: bytes, keyframe=True, timestamp=None) -> None:
        self.callback_on_frame(frame)


class ImageSourcePiCam(ImageSource):
    def __init__(self, output_width: int, on_image: Callable[[bytes], None]):
        super().__init__(output_width, on_image)

        self.camera: Picamera2 = Picamera2()

        configuration: str = self.camera.create_still_configuration(main={'size': (640, 640)})
        self.camera.start(configuration)

        self.encoder: JpegEncoder = JpegEncoder()
        self.encoder.output = CallbackOutput(on_image)
        
        self.camera.encode_stream_name = 'main'
        self.camera.start_encoder(self.encoder)
    
    def capture(self) -> None:
        # TODO: Currently don't seem to have direct control over when the image is captured. Seems to just go...
        request = self.camera.capture_request()
        request.release()
    
    def close(self) -> None:
        self.camera.close()
        self.camera.stop_encoder()
