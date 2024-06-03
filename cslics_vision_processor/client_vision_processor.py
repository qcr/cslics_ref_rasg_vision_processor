#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-05-31

import os
from time import sleep
import numpy
from paho.mqtt.client import Client
from paho.mqtt.enums import CallbackAPIVersion
from cslics_common import cslics_mqtt
from cslics_vision_processor.imaging import ImageSource
from ultralytics import YOLO
from ultralytics.engine.results import Results

SOFTWARE_NAME: str = 'cslics_client_camera'
SOFTWARE_VERSION: str = 'v0.0'
SOFTWARE_TAG: str = f'{SOFTWARE_NAME} {SOFTWARE_VERSION}'

def get_unique_identifier() -> str:
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if not line.startswith('Serial'):
                    continue

                return line.split(':')[1].strip()
    except:
        pass

    # In the case we cannot find one, a random one will do
    import random
    return ''.join(random.choice('0123456789abcdef') for i in range(16))

class CslicsClient:
    def __init__(self):
        self.is_running: bool = True
        self.identifier: str = get_unique_identifier()

        self.model: YOLO = self.setup_model()

        self.image_source: ImageSource = self.setup_image_source()

        self.image_topic: str = cslics_mqtt.getTopicForCamera(self.identifier, cslics_mqtt.TOPIC_POSTFIX_THUMBNAIL)

        self.client = Client(CallbackAPIVersion.VERSION2, f'{SOFTWARE_NAME}.{self.identifier}')
        self.setup_mqtt()
    
    def setup_image_source(self) -> ImageSource:
        print('Setting up image source...')
        from cslics_vision_processor.imaging import ImageSourcePiCam
        return ImageSourcePiCam(640, self.process_image_neural, self.publish_thumbnail)
    
    def setup_model(self) -> YOLO:
        model_path: str = os.path.expanduser('~/cslics_ref/models/cslics_20240117_yolov8x_640p_amt_alor2000.pt')
        print(f'Loading model from: "{model_path}"')
        model: YOLO = YOLO(model_path)
        print('Fusing model...')
        model.fuse()
        print('Model load completed!')
        return model
    
    def publish_identifier(self) -> None:
        self.client.publish(cslics_mqtt.TOPIC_CAMERAS, self.identifier)

    def publish_thumbnail(self, frame: bytes) -> None:
        self.publish_identifier()

        print(f'{SOFTWARE_NAME}: Capture length: {len(frame)}. Publishing...')
        self.client.publish(self.image_topic, frame)
    
    def process_image_neural(self, frame: numpy.ndarray) -> None:
        print('Recieved a frame with shape:', frame.shape)
        results: Results = self.model(frame)[0]

        detections: int = results.boxes.xyxyn.shape[0]

        for i in range(detections):
            detection = results.boxes.xyxyn[i]
            print('Found something!', detection)

    def setup_mqtt(self) -> None:
        print(f'{SOFTWARE_NAME}: Connecting to MQTT broker...')

        connected: bool = False

        while self.is_running and not connected:
            try:
                self.client.connect(cslics_mqtt.MQTT_BROKER, cslics_mqtt.MQTT_BROKER_PORT)
                connected = True
            except:
                sleep(1.0)
        
        if connected:
            print(f'{SOFTWARE_NAME}: Connected to MQTT broker!')
            self.client.loop_start()
            self.publish_identifier()
    
    def loop(self) -> None:
        while self.is_running:
            print(f'{SOFTWARE_NAME}: Capturing image...')
            self.image_source.capture()
            sleep(5.0)
        
        self.image_source.close()


def main() -> None:
    client = CslicsClient()
    client.loop()


if __name__ == '__main__':
    main()
