#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-05-31

from typing import Callable
import numpy
from cslics_vision_processor.imaging import ImageSource
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import Output
from picamera2.request import CompletedRequest

class CallbackOutput(Output):
    def __init__(self, callback_on_frame_encoded: Callable[[bytes], None]):
        self.callback_on_frame_encoded: Callable[[bytes], None] = callback_on_frame_encoded

    def outputframe(self, frame: bytes, keyframe=True, timestamp=None) -> None:
        self.callback_on_frame_encoded(frame)


class ImageSourcePiCam(ImageSource):
    def __init__(self, output_width: int, callback_on_frame_raw: Callable[[numpy.ndarray], None], callback_on_frame_encoded: Callable[[bytes], None]):
        super().__init__(output_width, callback_on_frame_raw, callback_on_frame_encoded)

        self.camera: Picamera2 = Picamera2()

        # TODO: Should figure out how to get the ideal resolution from the camera (aspect ratio doesn't matter)
        configuration: str = self.camera.create_still_configuration(main={'size': (640, 480)})
        self.camera.configure(configuration)

        self.encoder: JpegEncoder = JpegEncoder()
        self.encoder.output = CallbackOutput(self.callback_on_frame_encoded)
        
        self.camera.encode_stream_name = 'main'
        self.camera.start_encoder(self.encoder)
    
    def capture(self) -> None:
        # If the camera is left on, it captures continuously
        self.camera.start()
        request: CompletedRequest = self.camera.capture_request()
        self.camera.stop()
        self.callback_on_frame_raw(request.make_array('main'))
        request.release()
    
    def close(self) -> None:
        self.camera.close()
        self.camera.stop_encoder()
