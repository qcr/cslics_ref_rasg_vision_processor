#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-05-31

import os, struct, sys, numpy
from enum import Enum
from typing import Optional, List, Tuple
from time import sleep
from paho.mqtt.client import Client
from paho.mqtt.enums import CallbackAPIVersion
from cslics_common import cslics_mqtt
from cslics_vision_processor.imaging import ImageSource
from ultralytics import YOLO
from ultralytics.engine.results import Results

SOFTWARE_NAME: str = 'cslics_client_camera'
SOFTWARE_VERSION: str = 'v0.0'
SOFTWARE_TAG: str = f'{SOFTWARE_NAME} {SOFTWARE_VERSION}'

class ImageSourceType(Enum):
    PICAM = 0
    STORAGE_LOCAL = 1

def get_unique_identifier() -> str:
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if not line.startswith('Serial'):
                    continue

                return line.split(':')[1].strip()
    except:
        pass

    print(f'{SOFTWARE_NAME}: Unable to find a source of unique ID... Generating one!')

    # In the case we cannot find one, a random one will do
    import random
    return ''.join(random.choice('0123456789abcdef') for i in range(16))

class CslicsArgs:
    image_source: ImageSourceType = ImageSourceType.PICAM
    image_directory: Optional[str] = None
    model_size: int = 640
    model_path: Optional[str] = None 

class CslicsClient:
    def __init__(self, args: CslicsArgs):
        self.is_running: bool = True
        self.identifier: str = get_unique_identifier()

        self.state: int = -1

        self.image_source: ImageSource = self.setup_image_source(args)
        self.model: YOLO = self.setup_model(args)

        self.topic_thumbnail: str = cslics_mqtt.getTopicForCamera(self.identifier, cslics_mqtt.TOPIC_POSTFIX_THUMBNAIL)
        self.topic_boxes: str = cslics_mqtt.getTopicForCamera(self.identifier, cslics_mqtt.TOPIC_POSTFIX_BOXES)
        self.topic_counts: str = cslics_mqtt.getTopicForCamera(self.identifier, cslics_mqtt.TOPIC_POSTFIX_LATEST_COUNTS)
        self.topic_state: str = cslics_mqtt.getTopicForCamera(self.identifier, cslics_mqtt.TOPIC_POSTFIX_STATE)

        self.client = Client(CallbackAPIVersion.VERSION2, f'{SOFTWARE_NAME}.{self.identifier}')
        self.setup_mqtt()

        self.update_state(cslics_mqtt.STATE_IDLE)
    
    def update_state(self, state: int) -> None:
        self.state = state
        self.client.publish(self.topic_state, state)
    
    def setup_image_source(self, args: CslicsArgs) -> ImageSource:
        print(f'{SOFTWARE_NAME}: Setting up image source...')
        
        if (args.image_source == ImageSourceType.STORAGE_LOCAL):
            from cslics_vision_processor.imaging import ImageSourceStorageLocal
            return ImageSourceStorageLocal(args.model_size, self.process_image_neural, self.publish_thumbnail, args.image_directory)
        
        if (args.image_source == ImageSourceType.PICAM):
            from cslics_vision_processor.imaging import ImageSourcePiCam
            return ImageSourcePiCam(args.model_size, self.process_image_neural, self.publish_thumbnail)
    
    def setup_model(self, args: CslicsArgs) -> YOLO:
        model_path: str = os.path.expanduser(args.model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Pre-trained model does not exist at path: {model_path}')
        
        print(f'{SOFTWARE_NAME}: Loading model from: "{model_path}"')
        model: YOLO = YOLO(model_path)
        print(f'{SOFTWARE_NAME}: Fusing model...')
        model.fuse()
        print(f'{SOFTWARE_NAME}: Model load completed!')
        
        return model
    
    def publish_identifier(self) -> None:
        self.client.publish(cslics_mqtt.TOPIC_CAMERAS, self.identifier)

    def publish_thumbnail(self, frame: bytes) -> None:
        self.publish_identifier()

        print(f'{SOFTWARE_NAME}: Capture length: {len(frame)}. Publishing...')
        self.client.publish(self.topic_thumbnail, frame)
    
    def process_image_neural(self, frame: numpy.ndarray) -> None:
        self.update_state(cslics_mqtt.STATE_PROCESSING)

        results: Results = self.model(frame)[0]
        result_count: int = len(results.boxes)

        print(f'{SOFTWARE_NAME}: Detected {result_count} corals!')

        boxes_buffer: bytearray = bytearray(result_count * cslics_mqtt.STRUCT_BOX_SIZE)

        for i in range(result_count):
            edges = results.boxes.xyxyn[i]
            label = results.boxes.cls[i].item()
            struct.pack_into(cslics_mqtt.STRUCT_BOX_FORMAT, boxes_buffer, i * cslics_mqtt.STRUCT_BOX_SIZE, edges[0], edges[1], edges[2], edges[3], int(label))
        
        self.client.publish(self.topic_counts, struct.pack(cslics_mqtt.STRUCT_COUNT_FORMAT, result_count))
        self.client.publish(self.topic_boxes, boxes_buffer)

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
            self.update_state(cslics_mqtt.STATE_IDLE)
            sleep(1.0)
            # TODO: Sleep until next image should be captured

            self.update_state(cslics_mqtt.STATE_PRE_IMAGING)
            sleep(1.0)

            print(f'{SOFTWARE_NAME}: Capturing image...')
            self.update_state(cslics_mqtt.STATE_IMAGING)
            self.image_source.capture()
        
        self.image_source.close()

def parse_arguments() -> Tuple[bool, CslicsArgs]:
    options: CslicsArgs = CslicsArgs()
    args: List[str] = list(sys.argv)

    while len(args) > 0:
        arg: str = args.pop(0).lower()
        
        if arg == '-h' or arg == '--help':
            print(f'{SOFTWARE_TAG}')
            print(f'-h, --help: Print this help text.')
            print(f'-c, --capture-type: [{ImageSourceType.PICAM.name}, {ImageSourceType.STORAGE_LOCAL.name}]')
            print(f'-m, --model-path: /path/to/model.pt')
            print(f'-s, --model-size: int')
            print(f'-d, --image-directory: /path/to/images/')
            
            return False, options

        if len(args) == 0:
            break

        if arg == '-c' or arg == '--capture-type':
            options.image_source = ImageSourceType[args.pop(0).upper()]
            continue

        if arg == '-m' or arg == '--model-path':
            options.model_path = args.pop(0)
            continue

        if arg == '-s' or arg == '--model-size':
            options.model_size = int(args.pop(0))
            continue
        
        if arg == '-d' or arg == '--image-directory':
            options.image_directory = args.pop(0)

    return (True, options)


def main() -> None:
    run, options = parse_arguments()

    if not run:
        return
    
    client = CslicsClient(options)
    client.loop()


if __name__ == '__main__':
    main()
