#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-05-31

import os, struct, sys, numpy
from enum import Enum
from typing import Optional, List
from time import sleep
from paho.mqtt.client import Client
from paho.mqtt.enums import CallbackAPIVersion
from cslics_common import comms
from cslics_common.comms import VisionProcessorState, Box
from cslics_vision_processor.imaging import ImageSource
from ultralytics import YOLO
from ultralytics.engine.results import Results

SOFTWARE_NAME: str = 'cslics_client_vision_processor'
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
    broker_host: Optional[str] = None
    broker_port: Optional[int] = None
    image_source: ImageSourceType = ImageSourceType.PICAM
    image_directory: Optional[str] = None
    model_path: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        directory_requirement: bool = (self.image_source == ImageSourceType.STORAGE_LOCAL) == (self.image_directory != None)
        return self.broker_host != None and\
               self.broker_port != None and\
               self.model_path != None and\
               directory_requirement


class CslicsClient:
    def __init__(self, options: CslicsArgs):
        self.is_running: bool = True
        self.identifier: str = get_unique_identifier()

        self.state: int = -1
        self.image_index: int = 0

        self.model: YOLO = self.setup_model(options)

        model_size: int = self.model.overrides['imgsz']

        self.image_source: ImageSource = self.setup_image_source(options, model_size)

        self.topic_thumbnail: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_THUMBNAIL)
        self.topic_boxes: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_BOXES)
        self.topic_counts: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_COUNTS)
        self.topic_state: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_STATE)

        self.client = Client(CallbackAPIVersion.VERSION2, f'{SOFTWARE_NAME}.{self.identifier}')
        self.setup_mqtt(options)

        self.update_state(VisionProcessorState.IDLE)
    
    def update_state(self, state: VisionProcessorState) -> None:
        self.state = state
        self.client.publish(self.topic_state, state.value, retain=True)
    
    def setup_image_source(self, args: CslicsArgs, model_size: int) -> ImageSource:
        print(f'{SOFTWARE_NAME}: Setting up image source...')
        
        if (args.image_source == ImageSourceType.STORAGE_LOCAL):
            from cslics_vision_processor.imaging import ImageSourceStorageLocal
            return ImageSourceStorageLocal(model_size, self.process_image_neural, self.publish_thumbnail, args.image_directory)
        
        if (args.image_source == ImageSourceType.PICAM):
            from cslics_vision_processor.imaging import ImageSourcePiCam
            return ImageSourcePiCam(model_size, self.process_image_neural, self.publish_thumbnail)
    
    def setup_model(self, options: CslicsArgs) -> YOLO:
        model_path: str = os.path.expanduser(options.model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Pre-trained model does not exist at path: {model_path}')
        
        print(f'{SOFTWARE_NAME}: Loading model from: "{model_path}"')
        model: YOLO = YOLO(model_path)
        print(f'{SOFTWARE_NAME}: Fusing model...')
        model.fuse()
        print(f'{SOFTWARE_NAME}: Model load completed!')
        
        return model
    
    def publish_identifier(self) -> None:
        self.client.publish(comms.TOPIC_CAMERAS, self.identifier)

    def publish_thumbnail(self, frame: bytes) -> None:
        self.publish_identifier()

        print(f'{SOFTWARE_NAME}: Capture length: {len(frame)}. Publishing...')
        self.client.publish(self.topic_thumbnail, comms.pack_image(self.image_index, frame))
    
    def process_image_neural(self, frame: numpy.ndarray) -> None:
        self.update_state(VisionProcessorState.PROCESSING)

        results: Results = self.model(frame)[0]
        result_count: int = len(results)
        counts: List[int] = []

        for i in range(len(self.model.names)):
            counts.append(0)

        print(f'{SOFTWARE_NAME}: Detected {result_count} corals!')

        def create_box(index: int) -> Box:
            label = int(results.boxes.cls[index].item())
            counts[label] += 1
            return Box(*results.boxes.xyxyn[index], label=label)
        
        self.client.publish(self.topic_boxes, comms.pack_boxes(self.image_index, result_count, create_box))
        self.client.publish(self.topic_counts, comms.pack_counts(self.image_index, counts))

    def setup_mqtt(self, options: CslicsArgs) -> None:
        print(f'{SOFTWARE_NAME}: Connecting to MQTT broker at {options.broker_host}:{options.broker_port}...')

        connected: bool = False

        while self.is_running and not connected:
            try:
                self.client.connect(options.broker_host, options.broker_port)
                connected = True
            except:
                sleep(1.0)
        
        if connected:
            print(f'{SOFTWARE_NAME}: Connected to MQTT broker!')
            self.client.loop_start()
            self.publish_identifier()
    
    def loop(self) -> None:
        while self.is_running:
            self.update_state(VisionProcessorState.IDLE)
            sleep(1.0)
            # TODO: Sleep until next image should be captured

            self.update_state(VisionProcessorState.PRE_IMAGING)
            sleep(1.0)

            self.image_index += 1
            print(f'{SOFTWARE_NAME}: Capturing image...')
            self.update_state(VisionProcessorState.IMAGING)
            self.image_source.capture()
        
        self.image_source.close()

def print_help() -> None:
    print(SOFTWARE_TAG)
    print('-h, --help: Print this help text.')
    print(f'-c, --capture-type: [{ImageSourceType.PICAM.name}, {ImageSourceType.STORAGE_LOCAL.name}]')
    print('\nThe following arguments are required!\n')
    print('--host: [IP address or domain of MQTT broker host]')
    print('--port: [Port of MQTT broker]')
    print('-m, --model-path: /path/to/model.pt')
    print('-d, --image-directory: /path/to/images/')
    print(f'\nNOTE: Directory only required when using {ImageSourceType.STORAGE_LOCAL.name}')

def parse_arguments() -> CslicsArgs:
    options: CslicsArgs = CslicsArgs()
    args: List[str] = list(sys.argv)

    while len(args) > 0:
        arg: str = args.pop(0).lower()
        
        if arg == '-h' or arg == '--help':
            return options

        if len(args) == 0:
            break

        if arg == '--host':
            options.broker_host = args.pop(0)
            continue

        if arg == '--port':
            options.broker_port = int(args.pop(0))
            continue

        if arg == '-c' or arg == '--capture-type':
            options.image_source = ImageSourceType[args.pop(0).upper()]
            continue

        if arg == '-m' or arg == '--model-path':
            options.model_path = args.pop(0)
            continue
        
        if arg == '-d' or arg == '--image-directory':
            options.image_directory = args.pop(0)
            continue

    return options


def main() -> None:
    options: CslicsArgs = parse_arguments()

    if not options.is_valid:
        print_help()
        return
    
    client = CslicsClient(options)
    client.loop()


if __name__ == '__main__':
    main()
