#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-05-31

import time
import os, sys, numpy, logging
from typing import Optional, List
from enum import Enum
from argparse import ArgumentParser, ArgumentError
from logging import Logger
from paho.mqtt.client import Client, MQTTMessage
from paho.mqtt.enums import CallbackAPIVersion
from cslics_common import comms
from cslics_common.comms import VisionProcessorState, VisionProcessorMode, Box
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
        parser.add_argument('--config_file_path', required=False, default=None, help=f'The file path to the camera configuration YAML file')

        args = parser.parse_args()

        self.broker_host = args.broker_host
        self.broker_port = args.broker_port
        self.model_path = args.model_path
        self.image_source = ImageSourceType[args.image_source]
        self.image_directory = args.image_directory
        self.identifier = args.identifier
        self.config_path = args.config_file_path

        if self.image_source is ImageSourceType.STORAGE_LOCAL and self.image_directory == None:
            raise ArgumentError(arg_image_directory, f'Argument must be specified when using image_source {ImageSourceType.STORAGE_LOCAL.name}!')

    @property
    def is_valid(self) -> bool:
        directory_requirement: bool = (self.image_source == ImageSourceType.STORAGE_LOCAL) == (self.image_directory != None)
        return self.broker_host != None and\
               self.broker_port != None and\
               self.model_path != None and\
               directory_requirement

##
# @brief class CslicsClient - The MQTT message state machine for vision processing.
class CslicsClient:


    def __init__(self, options: CslicsArgs, logger: Logger):
        self.logger: Logger = logger.getChild(CslicsClient.__name__)
        self.is_running: bool = True
        self.identifier: str = self.get_unique_identifier(options)
        # set the configuration file path
        self.config_path: str = options.config_path 

        self.state: int = -1
        self.mode: int = VisionProcessorMode.LAZY.value
        self.image_index: int = 0

        self.model: YOLO = self.setup_model(options)

        model_size: int = self.model.overrides['imgsz']

        self.image_source: ImageSource = self.setup_image_source(options, model_size)
        # publish topics
        self.topic_thumbnail: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_THUMBNAIL)
        self.topic_boxes: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_BOXES)
        self.topic_counts: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_COUNTS)
        self.topic_state: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_STATE)

        #subscribe topics
        self.topic_trigger: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_CAMERA_TRIGGER)
        self.topic_settings: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_CAMERA_CONFIG)
        self.topic_mode: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_CAMERA_MODE)

        self.client = Client(CallbackAPIVersion.VERSION2, f'{SOFTWARE_NAME}.{self.identifier}')
        self.setup_mqtt(options)
         # set the message callback function
        self.client.on_message = self.on_message
        # set the initial camera state to Idle
        self.update_state(VisionProcessorState.IDLE)
        # setup subscriptions: for camera triggers
        self.client.subscribe(self.topic_trigger)
         # setup subscriptions: for camera configuration
        self.client.subscribe(self.topic_settings)
         # setup subscriptions: for mode of camera operations
        self.client.subscribe(self.topic_mode)

    ##
    # @brief on_message - the MQTT subscribed messae callback function
    # @param client : the client instance for this callback
    # @param userdata : the private user data as set in Client() or user_data_set()
    # @param message (MQTTMessage) – the received message. This is a class with members topic, payload, qos, retain.
    def on_message(self, client: Client, userdata, message: MQTTMessage):
        print(message.topic, message.payload)
        # Select the topic action
        if message.topic == self.topic_trigger:
            pass
        elif message.topic == self.topic_settings:
            # make sure we are processing this message in the correct camera mode
            if self.mode == VisionProcessorMode.FOCUS_ADJUST.value:
                # get the payload as byte array
                msg = list(message.payload)
                # get the focus value
                foc = int(msg[1])
                # set the focus
                self.image_source.set_focus([foc])

        elif message.topic == self.topic_mode:
            # get the message asa string
            msg = str(message.payload, "utf-8")
            # check the message is digit
            if msg.isdigit():
                # get the mode
                the_mode = int(msg)
                # make sure the mode we received is within range
                if VisionProcessorMode.LAZY.value <= the_mode <= VisionProcessorMode.FOCUS_ADJUST.value:
                    # set the mode
                    self.mode = the_mode

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
            return ImageSourcePiCam(model_size, self.process_image_neural, self.publish_thumbnail, self.logger, self.config_path)
    
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
        self.logger.info(f'Capture length (bytes): {len(frame)}. Publishing...')
        self.client.publish(self.topic_thumbnail, comms.pack_image(self.image_index, frame))
        # print(self.image_source.get_focus())
        # do mqtt messages
        # self.client.loop_misc()
    
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
                time.sleep(1.0)
        
        if connected:
            self.logger.info('Connected to MQTT broker!')
            # self.client.loop_start()
            self.publish_identifier()
    
    def loop(self) -> None:
        # get the current time in seconds
        t0 = time.time()
        # the program loop
        while self.is_running:
            # reset the 1-seond signal
            is_1sec = False
            # get the current time running
            t1 = time.time()
            # compute a delay test
            if (t1 - t0) >= 1.0:
                # signal 1-second events
                is_1sec = True
                # restart the stop watch
                t0 = t1
            # give this program loop a 1 second period for ids and states
            time.sleep(1)
            # publish the device identifier
            self.publish_identifier()
            # do mqtt message reads
            self.client.loop_read()
            # if there are messages to write
            if self.client.want_write():
                # write messages
                self.client.loop_write()
            # define the Modes
            if self.mode == VisionProcessorMode.LAZY.value:
                if is_1sec:
                    self.update_state(VisionProcessorState.IDLE)
            elif self.mode == VisionProcessorMode.MONITORING.value:
                if is_1sec:
                    self.update_state(VisionProcessorState.IDLE)
                    time.sleep(1.0)
                    # TODO: Sleep until next image should be captured

                    self.update_state(VisionProcessorState.PRE_IMAGING)
                    time.sleep(1.0)

                    self.image_index += 1
                    self.logger.info('Capturing image...')
                    self.update_state(VisionProcessorState.IMAGING)
                    self.image_source.capture()
            elif self.mode == VisionProcessorMode.SCIENCE.value:
                pass
            elif self.mode == VisionProcessorMode.FOCUS_ADJUST.value:
                if is_1sec:
                    # publish the focus mode
                    self.update_state(VisionProcessorState.FOCUS_ADJUST)
                # start the camera thumbnail stream
                self.image_source.start()
        # close the image source
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
