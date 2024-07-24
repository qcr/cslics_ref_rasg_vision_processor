#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-05-31

import os, sys, numpy, logging
from typing import Optional, List
from enum import Enum
from argparse import ArgumentParser, ArgumentError
from time import sleep
from logging import Logger
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


class CslicsArgs:
    broker_host: Optional[str] = None
    broker_port: Optional[int] = None
    model_path: Optional[str] = None
    image_source: ImageSourceType = ImageSourceType.PICAM
    image_directory: Optional[str] = None
    identifier: Optional[str] = None

    def __init__(self):
        parser: ArgumentParser = ArgumentParser(SOFTWARE_TAG, description='CSLICS client for Luxonis OAK PoE compatible edge computing devices')
        parser.add_argument('broker_host', metavar='host', default='localhost', help='URI for the MQTT broker host')
        parser.add_argument('broker_port', metavar='port', default=1883, type=int, help='Port for the MQTT broker host')
        parser.add_argument('model_path', metavar='/path/to/model.pt', help='Path to the model file to be used on the CSLICS Vision Processor')
        parser.add_argument('--image-source', required=False, default=ImageSourceType.PICAM, choices=ImageSourceType.__members__, help='Source to use for acquiring images')
        arg_image_directory = parser.add_argument('--image-directory', required=False, help=f'Directory to use for images if {ImageSourceType.STORAGE_LOCAL.name} is the selected image source')
        parser.add_argument('-id', '--identifier', required=False, help=f'Override the identifier discovery')

        args = parser.parse_args()

        self.broker_host = args.broker_host
        self.broker_port = args.broker_port
        self.model_path = args.model_path
        self.image_source = ImageSourceType[args.image_source]
        self.image_directory = args.image_directory
        self.identifier = args.identifier

        if self.image_source is ImageSourceType.STORAGE_LOCAL and self.image_directory == None:
            raise ArgumentError(arg_image_directory, f'Argument must be specified when using image_source {ImageSourceType.STORAGE_LOCAL.name}!')

    @property
    def is_valid(self) -> bool:
        directory_requirement: bool = (self.image_source == ImageSourceType.STORAGE_LOCAL) == (self.image_directory != None)
        return self.broker_host != None and\
               self.broker_port != None and\
               self.model_path != None and\
               directory_requirement


class CslicsClient:
    def __init__(self, options: CslicsArgs, logger: Logger):
        self.logger: Logger = logger.getChild(CslicsClient.__name__)
        self.is_running: bool = True
        self.identifier: str = self.get_unique_identifier(options)

        self.state: int = -1
        self.image_index: int = 0

        self.model: YOLO = self.setup_model(options)

        model_size: int = self.model.overrides['imgsz']

        self.image_source: ImageSource = self.setup_image_source(options, model_size)

        self.topic_thumbnail: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_THUMBNAIL)
        self.topic_boxes: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_BOXES)
        self.topic_counts: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_COUNTS)
        self.topic_state: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_STATE)

        # TODO: This is not robust to the MQTT broker disconnecting
        self.client = Client(CallbackAPIVersion.VERSION2, f'{SOFTWARE_NAME}.{self.identifier}')
        self.setup_mqtt(options)

        self.update_state(VisionProcessorState.IDLE)

    def get_unique_identifier(self, options: CslicsArgs) -> str:
        if options.identifier is not None:
            self.logger.warning(f'Using override identifier: {options.identifier}')
            return options.identifier
        
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if not line.startswith('Serial'):
                        continue

                    return line.split(':')[1].strip()
        except:
            pass

        self.logger.warning('Unable to find a source of unique ID... Generating one!')

        # In the case we cannot find one, a random one will do
        import random
        return ''.join(random.choice('0123456789ABCDEF') for i in range(16))
    
    def update_state(self, state: VisionProcessorState) -> None:
        self.state = state
        self.client.publish(self.topic_state, state.value, retain=True)
    
    def setup_image_source(self, args: CslicsArgs, model_size: int) -> ImageSource:
        self.logger.info('Setting up image source...')
        
        if (args.image_source == ImageSourceType.STORAGE_LOCAL):
            from cslics_vision_processor.imaging import ImageSourceStorageLocal
            return ImageSourceStorageLocal(model_size, self.process_image_neural, self.publish_thumbnail, args.image_directory, self.logger)
        
        if (args.image_source == ImageSourceType.PICAM):
            from cslics_vision_processor.imaging import ImageSourcePiCam
            return ImageSourcePiCam(model_size, self.process_image_neural, self.publish_thumbnail, self.logger)
    
    def setup_model(self, options: CslicsArgs) -> YOLO:
        model_path: str = os.path.expanduser(options.model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Pre-trained model does not exist at path: {model_path}')
        
        self.logger.info(f'Loading model from: "{model_path}"')
        model: YOLO = YOLO(model_path)
        self.logger.info('Fusing model...')
        model.fuse()
        self.logger.info('Model load completed!')
        
        return model
    
    def publish_identifier(self) -> None:
        self.client.publish(comms.TOPIC_CAMERAS, self.identifier)

    def publish_thumbnail(self, frame: bytes) -> None:
        self.publish_identifier()

        self.logger.info(f'Capture length (bytes): {len(frame)}. Publishing...')
        self.client.publish(self.topic_thumbnail, comms.pack_image(self.image_index, frame))
    
    def process_image_neural(self, frame: numpy.ndarray) -> None:
        self.update_state(VisionProcessorState.PROCESSING)

        results: Results = self.model(frame)[0]
        result_count: int = len(results)
        counts: List[int] = []

        for i in range(len(self.model.names)):
            counts.append(0)

        self.logger.info(f'Detected {result_count} corals!')

        def create_box(index: int) -> Box:
            label = int(results.boxes.cls[index].item())
            counts[label] += 1
            return Box(*results.boxes.xyxyn[index], label=label)
        
        self.client.publish(self.topic_boxes, comms.pack_boxes(self.image_index, result_count, create_box))
        self.client.publish(self.topic_counts, comms.pack_counts(self.image_index, counts))

    def setup_mqtt(self, options: CslicsArgs) -> None:
        self.logger.info(f'Connecting to MQTT broker at {options.broker_host}:{options.broker_port}...')

        connected: bool = False

        while self.is_running and not connected:
            try:
                self.client.connect(options.broker_host, options.broker_port)
                connected = True
            except:
                sleep(1.0)
        
        if connected:
            self.logger.info('Connected to MQTT broker!')
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
            self.logger.info('Capturing image...')
            self.update_state(VisionProcessorState.IMAGING)
            self.image_source.capture()
        
        self.image_source.close()


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    logger: Logger = logging.getLogger(SOFTWARE_NAME)

    logger.info(f'Starting {SOFTWARE_TAG}')

    options: CslicsArgs = CslicsArgs()

    if not options.is_valid:
        return
    
    client = CslicsClient(options, logger)
    client.loop()


if __name__ == '__main__':
    main()
