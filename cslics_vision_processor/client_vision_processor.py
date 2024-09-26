#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-05-31

import os, numpy, logging, json, time, cv2
from typing import Optional, List
from enum import Enum
from argparse import ArgumentParser, ArgumentError
from pathlib import Path
from logging import Logger
from paho.mqtt.client import Client, MQTTMessage
from paho.mqtt.enums import CallbackAPIVersion
from cslics_common import comms
from cslics_common.comms import VisionProcessorState, VisionProcessorMode
from cslics_vision_processor.imaging import ImageSource
from ultralytics import YOLO
from ultralytics.engine.results import Results
from cslics_vision_processor.imaging import ImageSourceStorageLocal

SOFTWARE_NAME: str = 'cslics_client_vision_processor'
SOFTWARE_VERSION: str = 'v0.0'
SOFTWARE_TAG: str = f'{SOFTWARE_NAME} {SOFTWARE_VERSION}'

# the method being used to down-sample the raw frame image for ML
CAPTURE_DOWNSAMPLE_METHOD = cv2.INTER_AREA


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
        parser: ArgumentParser = ArgumentParser(SOFTWARE_TAG, description='CSLICS client for edge computing devices')
        parser.add_argument('broker_host', metavar='host', default='localhost', help='URI for the MQTT broker host')
        parser.add_argument('broker_port', metavar='port', default=1883, type=int, help='Port for the MQTT broker host')
        parser.add_argument('model_path', metavar='/path/to/model.pt', help='Path to the model file to be used on the CSLICS Vision Processor')
        parser.add_argument('--image-source', required=False, default=ImageSourceType.PICAM, choices=ImageSourceType.__members__, help='Source to use for acquiring images')
        arg_image_directory = parser.add_argument('--image-directory', required=False, help=f'Directory to use for images if {ImageSourceType.STORAGE_LOCAL.name} is the selected image source')
        parser.add_argument('-id', '--identifier', required=False, help=f'Override the identifier discovery')
        parser.add_argument('--config_file_path', required=False, default=None, help=f'The file path to the camera configuration JSON file')
        parser.add_argument('--persist_path', required=False, default=None, help=f'The file path to the camera persistant settings file')
        parser.add_argument('--process_conf_path', required=False, default=None, help=f'The file path to the camera process configuration JSON file')

        args = parser.parse_args()

        self.broker_host = args.broker_host
        self.broker_port = args.broker_port
        self.model_path = args.model_path
        self.image_source = ImageSourceType[args.image_source]
        self.image_directory = args.image_directory
        self.identifier = args.identifier
        self.config_path = args.config_file_path
        self.persist_path = args.persist_path
        self.process_path = args.process_conf_path

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
        self.persist_path: str = self.options.persist_path
        self.process_path: str = self.options.process_path

        # process configuration parameters
        self.HEARTBEAT_RATE: float = 1.0
        self.MONITOR_IDLE_TIME: float = 1.0
        self.MONITOR_PRE_TIME: float = 1.0
        self.MONITOR_CAPTURE_TIME: float = 1.0 # this duration is added to the time it takes to capture and ML count
        self.LAZY_MODE_FRAME_WAIT: float = 10.0
        self.FOCUS_MODE_FRAME_WAIT: float = 0.2
        self.SCIENCE_MODE_TIME: float = 1800.0
        self.PERSISTED_WRITE_TIME: float = 60.0 # every five minutes
        self.LOOP_RATE = 20.0 # Rate float in Hz 

        # if given an existing path, otherwise just use default
        if os.path.exists(self.process_path):
            # open the persisted configuration (in binary)
            with open(self.process_path, 'r') as process_file:
                # load the JSON
                conf = json.load(process_file)
                self.HEARTBEAT_RATE = float(conf["HEARTBEAT_RATE"])
                self.MONITOR_IDLE_TIME = float(conf["MONITOR_IDLE_TIME"])
                self.MONITOR_PRE_TIME = float(conf["MONITOR_PRE_TIME"])
                self.MONITOR_CAPTURE_TIME = float(conf["MONITOR_CAPTURE_TIME"]) # this duration is added to the time it takes to capture and ML count
                self.LAZY_MODE_FRAME_WAIT = float(conf["LAZY_MODE_FRAME_WAIT"])
                self.FOCUS_MODE_FRAME_WAIT = float(conf["FOCUS_MODE_FRAME_WAIT"])
                self.SCIENCE_MODE_TIME = float(conf["SCIENCE_MODE_TIME"])
                self.PERSISTED_WRITE_TIME = float(conf["PERSISTED_WRITE_TIME"]) # every five minutes
                self.LOOP_RATE = float(conf["LOOP_RATE"]) # Rate float in Hz 
                # close the file
                process_file.close()

        self.state: int = -1
        self.mode: int = VisionProcessorMode.LAZY.value
        self.science_mode: bool = False
        self.science_mode_time: int = 0
        self.image_index: int = 0
        self.trigger_on: bool = False

        # a variable to emmit a thumbnail
        self.do_thumbnail: bool = False
        # the previous recieved settings
        self.previous_settings: bytes = None
        # the currently recieved settings
        self.current_settings: bytes = None
        # can persist settings
        self.can_persist = os.path.exists(self.persist_path)
        # if we have a valid file name for persisting camera settings
        if self.can_persist:
            # open the persisted configuration (in binary)
            with open(self.persist_path, 'rb') as persist_file:
                # get the persisted setting
                self.current_settings = persist_file.read()
                # close the file
                persist_file.close()

        # set up the YOLO model
        self.model: YOLO = self.setup_model(self.options)
        # get the default model directory path
        self.model_dir_path = os.path.dirname(os.path.expanduser(self.options.model_path))
         # the previous recieved settings
        self.previous_model_msg: bytes = None
        # the currently recieved settings
        self.current_model_msg: bytes = None

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
        self.topic_trigger: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_TRIGGER)
        self.topic_settings: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_CAMERA_CONFIG)
        self.topic_mode: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_MODE)
        self.topic_model: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_MODEL)
        self.topic_science: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_SCIENCE)

        # TODO: This is not robust to the MQTT broker disconnecting
        self.client = Client(CallbackAPIVersion.VERSION2, f'{SOFTWARE_NAME}.{self.identifier}')

        # set the MQTT message callback functions
        self.client.on_message = self.on_message
        self.client.on_connect = self.on_connect
        # initialise the MQTT client
        self.setup_mqtt(self.options)

    ##
    # @brief on_connect - the MQTT client callback function that gets called when the MQTT client successfully 
    # connects. This method sets up the MQTT subscription topics
    #     
    def on_connect(self, client, userdata, flags, reason_code, properties):
        # set the initial camera state to Idle
        self.update_state(VisionProcessorState.IDLE)
        # setup subscriptions: for camera triggers
        self.client.subscribe(self.topic_trigger)
        # setup subscriptions: for camera configuration
        self.client.subscribe(self.topic_settings)
        # setup subscriptions: for mode of camera operations
        self.client.subscribe(self.topic_mode)
        # setup subscription for the model
        self.client.subscribe(self.topic_model)
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
                # set the current setting string
                self.current_settings = message.payload
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
                        # turn off science mode
                        self.science_mode = False
                        self.update_state(VisionProcessorState.IDLE)
                    elif self.mode == VisionProcessorMode.MONITORING.value:
                        # make sure the camera is stopped
                        self.image_source.stop()
                        # set the state
                        self.update_state(VisionProcessorState.IDLE)
                    elif self.mode == VisionProcessorMode.FOCUS_ADJUST.value:
                        # make sure the camera is stopped
                        self.image_source.stop()
                        # turn off science mode
                        self.science_mode = False
                        # publish once the focus adjust state
                        self.update_state(VisionProcessorState.FOCUS_ADJUST)
        elif message.topic == self.topic_model:
            # set the model ,message
            self.current_model_msg = message.payload   
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
    
    ##
    # @brief update_model - given a model message, updates the YOLO model
    # @param message : the model message object
    def update_model(self, message: comms.ModelMessage) -> None:
        # contruct the path
        model_path: str = os.path.join(self.model_dir_path, message.name + ".pt")
        # setup the model given the path string
        the_model: YOLO = self.setup_model_from_path(model_path)
        # if the model loading was successful
        if the_model is not None:
            # set the model
            self.model = the_model
            self.model.conf = message.confidence_threshold
            self.model.iou = message.iou

    ##
    # @brief update_state - used to update and publishes the camera states when the camera is in Monitor mode.
    # @param state : the camera state     
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
    
    
    ##
    # @brief setup_model_from_path - given a path string, this method loads a YOLO model and runs fuse
    # @param model_path : the file path to the YOLO model
    # @return YOLO : the model on succcess, otherwise None
    def setup_model_from_path(self, model_path: str) -> YOLO:
        # if the file does not exists
        if not os.path.exists(model_path):
            return None
        # load the model
        self.logger.info(f'Loading model from: "{model_path}"')
        model: YOLO = YOLO(model_path)
        self.logger.info('Fusing model...')
        model.fuse()
        self.logger.info('Model load completed!')
        # return the model
        return model
    
    ##
    # @brief setup_model - open a YOLO model given a model path in a set of options.
    # @options : the set of options
    # @return YOLO - the model
    def setup_model(self, options: CslicsArgs) -> YOLO:
        # expand the path
        model_path: str = os.path.expanduser(options.model_path)
        # if the file does not exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Pre-trained model does not exist at path: {model_path}')
        # setup the model from path
        return self.setup_model_from_path(model_path)
    
    
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
            # update the image index
            self.image_index += 1
        # restore the do thumbnail state
        self.do_thumbnail = False

    def process_image_neural(self, frame: numpy.ndarray) -> None:
        # encode the frame as JPEG
        _, jpg_img = cv2.imencode('.jpeg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # pack the image
        buf: bytearray = comms.pack_image(self.image_index, jpg_img.tobytes())
        self.logger.info(f'Process neural image length (bytes): {len(buf)}. Publishing...')
        # if in science mode
        if self.science_mode:
            # publish the bytes
            self.client.publish(self.topic_science_data, buf)
            # if there are messages to write
            if self.client.want_write():
                # write messages
                self.client.loop_write()
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
             # encode the frame as JPEG
            _, jpg_img2 = cv2.imencode('.jpeg', cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
            # pack the image
            buf2: bytearray = comms.pack_image(self.image_index, jpg_img2.tobytes())
            self.logger.info(f'Process neural image length (bytes): {len(buf2)}. Publishing...')
            # publish the bytes
            self.client.publish(self.topic_thumbnail, buf2)
            # if there are messages to write
            if self.client.want_write():
                # write messages
                self.client.loop_write()
        else:
            # publish the bytes
            self.client.publish(self.topic_thumbnail, buf)
            # the frame is already the right size
            new_frame = frame
            # if there are messages to write
            if self.client.want_write():
                # write messages
                self.client.loop_write()

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

        boxes: List[comms.Box] = []

        for i in range(result_count):
            label = int(results.boxes.cls[i].item())
            counts[label] += 1
            boxes.append(comms.Box(*results.boxes.xyxyn[i], label=label))
        
        # include volume calc
        sampled_volume: float = self.image_source.get_dof_volume()
        print("Volume mm^3 ", sampled_volume)
        
        self.client.publish(self.topic_boxes, comms.BoxesMessage(self.image_index, sampled_volume, boxes).pack())
        self.client.publish(self.topic_counts, comms.CountsMessage(self.image_index, sampled_volume, counts).pack())
        # update the image index
        self.image_index += 1
        # do mqtt message reads
        self.client.loop_read()
        # if there are messages to write
        if self.client.want_write():
            # write messages
            self.client.loop_write()
    
    def cache_results(self, volume: float, results: Results) -> None:
        cache_path: Optional[Path] = self.try_get_cache_result_path()

        if cache_path is None:
            return
        
        result_count: int = len(results)

        boxes: List[comms.Box] = []
        counts: List[int] = []

        for i in range(len(self.model.names)):
            counts.append(0)

        for i in range(result_count):
            label = int(results.boxes.cls[i].item())
            counts[label] += 1
            boxes.append({
                'xyxyn': [float(tensor) for tensor in results.boxes.xyxyn[i]],
                'label': label
            })

        output: dict = {
            'volume': volume,
            'counts': counts,
            'boxes': boxes
        }

        with open(cache_path, 'w') as file:
            json.dump(output, file)
    
    def load_and_publish_cached_results(self) -> bool:
        cache_path: Optional[Path] = self.try_get_cache_result_path()

        if cache_path is None or not cache_path.exists():
            return False

        result: dict = {}

        try:
            with open(cache_path, 'r') as file:
                result = json.load(file)
        except:
            self.logger.error(f'Unable to read cached data! Path: {cache_path}')
            return

        if 'counts' not in result or 'boxes' not in result or 'volume' not in result:
            self.logger.error(f'Cached data exists but it malformed! Path: {cache_path}')
            return False

        boxes: List[comms.Box] = []

        try:
            for box_dict in result['boxes']:
                if 'xyxyn' not in box_dict or 'label' not in box_dict:
                    raise Exception('Box dict malformed!')

                boxes.append(comms.Box(*box_dict['xyxyn'], box_dict['label']))
        except:
            self.logger.error(f'Malformed box in cache file! Path: {cache_path}')
            return False
        
        volume: float = result['volume']

        self.client.publish(self.topic_boxes, comms.BoxesMessage(self.image_index, volume, boxes).pack())
        self.client.publish(self.topic_counts, comms.CountsMessage(self.image_index, volume, result['counts']).pack())

        return True
    
    def try_get_cache_result_path(self) -> Optional[Path]:
        if type(self.image_source) is not ImageSourceStorageLocal:
            return None
        
        image_path: Optional[Path] = self.image_source.latest_path
        
        if image_path is None:
            return None
        
        result_path: Path = image_path.parent.joinpath(f'{image_path.stem}.json')

        return result_path

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
        t0_persist = t0_id
        # set the loop rate as a wait time
        loop_rate = 1.0/self.LOOP_RATE
        # the program loop
        while self.is_running: 
            # sleep for 1/rate seconds
            time.sleep(loop_rate)
            # get the current time running
            t1 = time.time()
            # if time to publish a heartbeat
            if (t1 - t0_id) >= self.HEARTBEAT_RATE:
                # publish the device identifier
                self.publish_identifier()
                # restart the stop watch
                t0_id = t1
            # do mqtt message reads
            self.client.loop_read()
            # if there are messages to write
            if self.client.want_write():
                # write messages
                self.client.loop_write()
            # if time to write current setting
            if self.can_persist and (t1 - t0_persist) >= self.PERSISTED_WRITE_TIME:
                # update persist timer
                t0_persist = t1
                # open the persisted configuration (in binary)
                with open(self.persist_path, 'wb') as persist_file:
                    print("Writing to persist file")
                    # write the persisted setting
                    persist_file.write(self.current_settings) 
                    # close file
                    persist_file.close()

            # doing a science mode publish
            if self.science_mode:
                # if timed out
                if (t1 - self.science_mode_time) >= self.SCIENCE_MODE_TIME:
                    self.science_mode = False
            if self.previous_settings != self.current_settings:
                #try:
                # parse the settings
                settings: comms.CameraSettings = comms.CameraSettings.from_buffer(self.current_settings)
                # start the camera thumbnail stream
                self.image_source.start()
                # set camera
                self.image_source.set_settings(settings)
                #except:
                #    self.logger.error("comms.CameraSettings could not parse the camera settings message.")
                # reset change
                self.previous_settings = self.current_settings
            if self.previous_model_msg != self.current_model_msg:
                try:
                    # parse the model message
                    msg: comms.ModelMessage = comms.ModelMessage.from_buffer(self.current_model_msg)
                    # update the YOLO model
                    self.update_model(msg)
                except:
                    self.logger.error("comms.ModelMessage could not parse the model message.")
                # reset the state change
                self.previous_model_msg = self.current_model_msg
            # define the Modes
            if self.mode == VisionProcessorMode.LAZY.value:
                # start the camera thumbnail stream
                self.image_source.start()
                # if time to publish a thumbnail
                if (t1 - t0_mon) >= self.LAZY_MODE_FRAME_WAIT:
                    # update timer
                    t0_mon = t1
                    # trigger a do thumbnail
                    self.do_thumbnail = True
            elif self.mode == VisionProcessorMode.FOCUS_ADJUST.value:
                # start the camera thumbnail stream
                self.image_source.start()
                # if time to publish a thumbnail
                if (t1 - t0_mon) >= self.FOCUS_MODE_FRAME_WAIT:
                    # update timer
                    t0_mon = t1
                    # trigger a do thumbnail
                    self.do_thumbnail = True
            elif self.mode == VisionProcessorMode.MONITORING.value: 
                # if the capture has been triggered
                if self.trigger_on:
                    # test for state transitions
                    if self.state == VisionProcessorState.IDLE and (t1 - t0_mon) >= self.MONITOR_IDLE_TIME:
                        # update timer
                        t0_mon = t1
                        # update the state
                        self.update_state(VisionProcessorState.PRE_IMAGING)
                    elif self.state == VisionProcessorState.PRE_IMAGING and (t1 - t0_mon) >= self.MONITOR_PRE_TIME:
                        # update timer
                        t0_mon = t1
                        # update the state
                        self.update_state(VisionProcessorState.IMAGING)
                    elif self.state == VisionProcessorState.IMAGING and (t1 - t0_mon) >= self.MONITOR_CAPTURE_TIME:
                        # capture the image and perfrom ML count
                        self.image_source.capture(int(self.science_mode == True))
                        # return to Idle state
                        self.update_state(VisionProcessorState.IDLE)
                        # update timer
                        t0_mon = t1
                        # restore the trigger state
                        self.trigger_on = False
            # make sure we are still connected
            while not self.client.is_connected():
                 # try to reconnect
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
