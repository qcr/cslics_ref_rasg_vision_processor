#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-05-31

import time
import cv2
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
HEARTBEAT_RATE: float = 1.0
MONITOR_IDLE_TIME: float = 1.0
MONITOR_PRE_TIME: float = 1.0
MONITOR_CAPTURE_TIME: float = 1.0 # this duration is added to the time it takes to capture and ML count
LAZY_MODE_FRAME_WAIT: float = 10.0
FOCUS_MODE_FRAME_WAIT: float = 0.2
SCIENCE_MODE_TIME: float = 1800.0
# the method being used to sample the raw frame image down for ML
CAPTURE_DOWNSAMPLE_METHOD = cv2.INTER_AREA
LOOP_RATE = 20.0 # Rate float in Hz


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
        self.options: CslicsArgs = options
        self.identifier: str = self.get_unique_identifier(self.options)
        # set the configuration file path
        self.config_path: str = self.options.config_path 

        self.state: int = -1
        self.mode: int = VisionProcessorMode.LAZY.value
        self.science_mode: bool = False
        self.science_mode_time: int = 0
        self.image_index: int = 0
        self.trigger_on: bool = False

        # a variable to emmit a thumbnail
        self.do_thumbnail: bool = False

        self.model: YOLO = self.setup_model(self.options)

        self.model_size: int = self.model.overrides['imgsz']

        self.image_source: ImageSource = self.setup_image_source(self.options, self.model_size)
        # publish topics
        self.topic_thumbnail: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_THUMBNAIL)
        self.topic_boxes: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_BOXES)
        self.topic_counts: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_COUNTS)
        self.topic_state: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_STATE)
        self.topic_science_data: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_SCIENCE_DATA)
        self.topic_thumbnail_cb = self.topic_thumbnail

        #subscribe topics
        self.topic_trigger: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_CAMERA_TRIGGER)
        self.topic_settings: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_CAMERA_CONFIG)
        self.topic_mode: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_CAMERA_MODE)
        self.topic_science: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_CAMERA_SCIENCE)

        self.client = Client(CallbackAPIVersion.VERSION2, f'{SOFTWARE_NAME}.{self.identifier}')
        # go to loop switch
        self.got_to_loop = True
        # set the message callback function
        self.client.on_message = self.on_message
        self.client.on_connect = self.on_connect
        self.setup_mqtt(self.options)
        
    def on_connect(self, client, userdata, flags, reason_code, properties):
        # set the initial camera state to Idle
        self.update_state(VisionProcessorState.IDLE)
        # setup subscriptions: for camera triggers
        self.client.subscribe(self.topic_trigger)
        # setup subscriptions: for camera configuration
        self.client.subscribe(self.topic_settings)
        # setup subscriptions: for mode of camera operations
        self.client.subscribe(self.topic_mode)
        # setup subscriptions: for science mode of camera operations
        self.client.subscribe(self.topic_science)

        
    ##
    # @brief on_message - the MQTT subscribed message callback function
    # @param client : the client instance for this callback
    # @param userdata : the private user data as set in Client() or user_data_set()
    # @param message (MQTTMessage) – the received message. This is a class with members topic, payload, qos, retain.
    def on_message(self, client: Client, userdata, message: MQTTMessage):
        print(message.topic, message.payload)
        # Select the topic action
        if message.topic == self.topic_trigger:
            # if in monitoring mode
            if self.mode == VisionProcessorMode.MONITORING.value and not self.trigger_on:
                self.trigger_on = True
                self.update_state(VisionProcessorState.IDLE)
        elif message.topic == self.topic_settings:
            # make sure we are processing this message in the correct camera mode
            if self.mode == VisionProcessorMode.FOCUS_ADJUST.value:
                # set the settings
                settings = comms.CameraSettings.from_buffer(message.payload)
                # set camera
                self.image_source.set_settings(settings)
                time.sleep(0.1)
        elif message.topic == self.topic_mode:
            # get the message as a string
            msg = str(message.payload, "utf-8")
            # check the message is digit
            if msg.isdigit():
                # get the mode
                the_mode = int(msg)
                # if we are already in this mode
                if  self.mode == the_mode:
                    return
                # make sure the mode we received is within range
                if VisionProcessorMode.LAZY.value <= the_mode <= VisionProcessorMode.FOCUS_ADJUST.value:
                    # set the mode
                    self.mode = the_mode
                    # initial the state
                    if self.mode == VisionProcessorMode.LAZY.value:
                        # make sure the camera is stopped
                        self.image_source.stop()
                        self.update_state(VisionProcessorState.IDLE)
                    elif self.mode == VisionProcessorMode.MONITORING.value:
                        # make sure the camera is stopped
                        self.image_source.stop()
                        # set the state
                        self.update_state(VisionProcessorState.IDLE)
                    elif self.mode == VisionProcessorMode.FOCUS_ADJUST.value:
                        # make sure the camera is stopped
                        self.image_source.stop()
                        # publish once the focus adjust state
                        self.update_state(VisionProcessorState.FOCUS_ADJUST)
        elif message.topic == self.topic_science:
            # is in science mode
            self.science_mode = self.mode == VisionProcessorMode.MONITORING.value
            # if in science mode
            if self.science_mode:
                # get current moment for timeout
                self.science_mode_time = time.time()

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
        
        if args.image_source == ImageSourceType.STORAGE_LOCAL:
            from cslics_vision_processor.imaging import ImageSourceStorageLocal
            return ImageSourceStorageLocal(model_size, self.process_image_neural, self.publish_thumbnail, args.image_directory, self.logger)
        
        if args.image_source == ImageSourceType.PICAM:
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
        # publish the device identifier
        self.client.publish(comms.TOPIC_CAMERAS, self.identifier)

    ##
    # @brief publish_thumbnail - The callback method for live-view images
    # @param frame: the live-view thumbnail data
    def publish_thumbnail(self, frame: bytes) -> None:
        # if not doing a thumbnail
        if not self.do_thumbnail:
            return
        if self.mode == VisionProcessorMode.LAZY.value or self.mode == VisionProcessorMode.FOCUS_ADJUST.value:
            self.logger.info(f'Live-view length (bytes): {len(frame)}. Publishing...')
            buf: bytearray = comms.pack_image(self.image_index, frame)
            self.client.publish(self.topic_thumbnail_cb, buf)
            # make sure thumbs don't suffocate MQTT
            """
            # do mqtt message reads
            self.client.loop_read()
            # while there are messages to write
            while self.client.want_write():
                # write messages
                self.client.loop_write()
            """
        # restore the do thumbnail state
        self.do_thumbnail = False


    def process_image_neural(self, frame: numpy.ndarray) -> None:
        # if in science mode
        if self.science_mode:
            # encode the frame as JPEG
            _, jpg_img = cv2.imencode('.jpeg', frame)
            # pack the image
            buf: bytearray = comms.pack_image(self.image_index, jpg_img.tobytes())
            self.logger.info(f'Science Mode image length (bytes): {len(buf)}. Publishing...')
            # publish the bytes
            self.client.publish(self.topic_science_data, buf)
            # get the frame shape
            height, width, _ = frame.shape
            # get ration
            camera_ratio: float = height / width
            # the image size used for raw images in ML
            output_height: int = self.model_size
            output_width: int = self.model_size
            # scale the correct dimension
            if width > height:
                output_height = int(round(self.model_size * camera_ratio))
            elif height > width:
                output_width = int(round(self.model_size / camera_ratio))
            # print("ML frame ", output_width, output_height)
            # create the resized frame
            new_frame = cv2.resize(frame, dsize=(output_width, output_height), 
                                   interpolation=CAPTURE_DOWNSAMPLE_METHOD)
        else:
            # the frame is already the right size
            new_frame = frame

        # set the processing state
        self.update_state(VisionProcessorState.PROCESSING)
        print(new_frame.shape)
        # Set the model with the raw frame
        results: Results = self.model(new_frame)[0]
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
        # update the image index
        self.image_index += 1
        # do mqtt message reads
        self.client.loop_read()
        # while there are messages to write
        while self.client.want_write():
            # write messages
            self.client.loop_write()

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
        t0_id = time.time()
        t0_mon = t0_id
        # set the loop rate as a wait time
        loop_rate = 1.0/LOOP_RATE
        # the program loop
        while self.is_running and self.got_to_loop: 
            # sleep for 1/rate seconds
            time.sleep(loop_rate)
            # get the current time running
            t1 = time.time()
            # if time to publish a heartbeat
            if (t1 - t0_id) >= HEARTBEAT_RATE:
                # publish the device identifier
                self.publish_identifier()
                # restart the stop watch
                t0_id = t1
            # do mqtt message reads
            self.client.loop_read()
            # while there are messages to write
            while self.client.want_write():
                # write messages
                self.client.loop_write()
            # doing a science mode publish
            if self.science_mode:
                # if timed out
                if (t1 - self.science_mode_time) >= SCIENCE_MODE_TIME:
                    self.science_mode = False
            # define the Modes
            if self.mode == VisionProcessorMode.LAZY.value:
                # start the camera thumbnail stream
                self.image_source.start()
                # if time to publish a thumbnail
                if (t1 - t0_mon) >= LAZY_MODE_FRAME_WAIT:
                    # update timer
                    t0_mon = t1
                    # trigger a do thumbnail
                    self.do_thumbnail = True
            elif self.mode == VisionProcessorMode.FOCUS_ADJUST.value:
                # start the camera thumbnail stream
                self.image_source.start()
                # if time to publish a thumbnail
                if (t1 - t0_mon) >= FOCUS_MODE_FRAME_WAIT:
                    # update timer
                    t0_mon = t1
                    # trigger a do thumbnail
                    self.do_thumbnail = True
            elif self.mode == VisionProcessorMode.MONITORING.value: 
                # if the capture has been triggered
                if self.trigger_on:
                    # test for state transitions
                    if self.state == VisionProcessorState.IDLE and (t1 - t0_mon) >= MONITOR_IDLE_TIME:
                        # update timer
                        t0_mon = t1
                        # update the state
                        self.update_state(VisionProcessorState.PRE_IMAGING)
                    elif self.state == VisionProcessorState.PRE_IMAGING and (t1 - t0_mon) >= MONITOR_PRE_TIME:
                        # update timer
                        t0_mon = t1
                        # update the state
                        self.update_state(VisionProcessorState.IMAGING)
                    elif self.state == VisionProcessorState.IMAGING and (t1 - t0_mon) >= MONITOR_CAPTURE_TIME:
                        # capture the image and perfrom ML count
                        self.image_source.capture(int(self.science_mode == True))
                        # return to Idle state
                        self.update_state(VisionProcessorState.IDLE)
                        # update timer
                        t0_mon = t1
                        # restore the trigger state
                        self.trigger_on = False
            # make sure we are still connected
            self.got_to_loop = self.client.is_connected()
        # try to reconnect
        if not self.got_to_loop:
            self.setup_mqtt(self.options)



def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    logger: Logger = logging.getLogger(SOFTWARE_NAME)

    logger.info(f'Starting {SOFTWARE_TAG}')

    options: CslicsArgs = CslicsArgs()

    if not options.is_valid:
        return
    
    client = CslicsClient(options, logger)
    client.loop()
    client.image_source.close()

if __name__ == '__main__':
    main()
