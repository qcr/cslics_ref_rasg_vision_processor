#!/usr/bin/env python3

import os, numpy, logging, json, time, cv2, socket
from typing import Optional, List
from enum import Enum
from argparse import ArgumentParser, ArgumentError
from pathlib import Path
from logging import Logger
from paho.mqtt.client import Client, MQTTMessage
from paho.mqtt.enums import CallbackAPIVersion
from cslics_mqtt import comms
from cslics_mqtt.comms import VisionProcessorState, VisionProcessorMode
from cslics_vision_processor.imaging import ImageSource
from ultralytics import YOLO
from ultralytics.engine.results import Results

SOFTWARE_NAME: str = 'cslics_client_vision_processor'
SOFTWARE_VERSION: str = 'v1.2'
SOFTWARE_TAG: str = f'{SOFTWARE_NAME} {SOFTWARE_VERSION}'

# the method being used to down-sample the raw frame image for ML
CAPTURE_DOWNSAMPLE_METHOD = cv2.INTER_AREA

def get_ip_address() -> str:
    try:
        return [(s.connect(('255.255.0.0', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]
    except:
        return '127.0.0.1'


class ImageSourceType(Enum):
    PICAM = 0
    STORAGE_LOCAL = 1


class CslicsArgs:
    __broker_host: str = 'localhost'
    __broker_port: int = 1883
    __models_path: Path
    __image_source: ImageSourceType = ImageSourceType.PICAM
    __image_directory: Optional[Path] = None
    __config_path: Path = None
    __persist_path: Optional[Path] = None
    __process_path: Optional[Path] = None

    identifier: Optional[Path] = None

    def __init__(self):
        parser: ArgumentParser = ArgumentParser(SOFTWARE_TAG, description='CSLICS client for edge computing devices')
        parser.add_argument('-b', '--broker-host', metavar='host', default=self.__broker_host, help='URI for the MQTT broker host')
        parser.add_argument('-p', '--broker-port', metavar='port', default=self.__broker_port, type=int, help='Port for the MQTT broker host')
        parser.add_argument('-m', '--models-path', required=True, metavar='/path/to/model/directory/', help='Path to the model files to be used on the CSLICS Vision Processor')
        parser.add_argument('--image-source', required=False, default=self.__image_source, choices=ImageSourceType.__members__, help='Source to use for acquiring images')
        arg_image_directory = parser.add_argument('--image-directory', required=False, help=f'Directory to use for images if {ImageSourceType.STORAGE_LOCAL.name} is the selected image source')
        parser.add_argument('-id', '--identifier', required=False, help=f'Override the identifier discovery')
        parser.add_argument('--config_file_path', required=True, help=f'The file path to the camera configuration JSON file')
        parser.add_argument('--persist_path', required=False, default=None, help=f'The file path to the camera persistant settings file')
        parser.add_argument('--process_conf_path', required=False, default=None, help=f'The file path to the camera process configuration JSON file')

        args = parser.parse_args()

        self.__broker_host = args.broker_host
        self.__broker_port = args.broker_port
        self.__models_path = Path(args.models_path)
        self.__image_source = ImageSourceType[args.image_source]
        self.__image_directory = args.image_directory
        self.identifier = args.identifier
        self.__config_path = Path(args.config_file_path)
        self.__persist_path = Path(args.persist_path) if args.persist_path is not None else None
        self.__process_path = Path(args.process_conf_path) if args.process_conf_path is not None else None

        if self.image_source is ImageSourceType.STORAGE_LOCAL and self.image_directory == None:
            raise ArgumentError(arg_image_directory, f'Argument must be specified when using image_source {ImageSourceType.STORAGE_LOCAL.name}!')

    @property
    def broker_host(self) -> str:
        return self.__broker_host

    @property
    def broker_port(self) -> int:
        return self.__broker_port

    @property
    def models_path(self) -> Path:
        return self.__models_path

    @property
    def image_source(self) -> ImageSourceType:
        return self.__image_source

    @property
    def image_directory(self) -> Optional[str]:
        return self.__image_directory

    @property
    def config_path(self) -> Path:
        return self.__config_path

    @property
    def persist_path(self) -> Optional[Path]:
        return self.__persist_path

    @property
    def process_path(self) -> Optional[Path]:
        return self.__process_path


class LoadedModel:
    __name: str
    __model: YOLO
    __size: int
    __is_fused: bool = False
    confidence_threshold: float
    iou: float

    def __init__(self, name: str, model: YOLO, confidence_threshold: float = 0.7, iou: float = 0.5):
        self.__name = name
        self.__model = model
        self.__size = self.__model.overrides['imgsz']
        self.confidence_threshold = confidence_threshold
        self.iou = iou

    @property
    def name(self) -> str:
        return self.__name
        
    @property
    def model(self) -> YOLO:
        return self.__model

    @property
    def size(self) -> int:
        return self.__size

    @property
    def is_fused(self) -> bool:
        return self.__is_fused

    def fuse(self) -> None:
        self.__model.fuse()
        self.__is_fused = True

    def process(self, source: numpy.ndarray, **model_kwargs) -> Results:
        if not self.is_fused:
            self.fuse()

        return self.__model(source, conf=self.confidence_threshold, iou=self.iou, **model_kwargs)[0]


##
# @brief class CslicsClient - The MQTT message state machine for vision processing.
class CslicsClient:
    def __init__(self, options: CslicsArgs, logger: Logger):
        self.logger: Logger = logger.getChild(CslicsClient.__name__)
        self.is_running: bool = True
        self.options: CslicsArgs = options
        self.identifier: str = self.get_unique_identifier(self.options)

        # process configuration parameters
        self.heartbeat_rate: float = 5.0
        self.monitor_idle_time: float = 1.0
        self.monitor_pre_time: float = 1.0
        self.monitor_capture_time: float = 1.0 # this duration is added to the time it takes to capture and ML count
        self.lazy_mode_frame_wait: float = 10.0
        self.focus_mode_frame_wait: float = 0.2
        self.science_mode_time: float = 1800.0
        self.persisted_write_time: float = 60.0
        self.loop_rate = 20.0 # Rate float in Hz 

        # if given an existing path, otherwise just use default
        if self.options.process_path is not None:
            if not self.options.process_path.exists():
                self.logger.warning('Process configuration path specified but does not exist!')
            else:
                # open the persisted configuration (in binary)
                with open(self.options.process_path, 'r') as process_file:
                    # load the JSON
                    conf = json.load(process_file)
                    self.heartbeat_rate = float(conf["HEARTBEAT_RATE"])
                    self.monitor_idle_time = float(conf["MONITOR_IDLE_TIME"])
                    self.monitor_pre_time = float(conf["MONITOR_PRE_TIME"])
                    self.monitor_capture_time = float(conf["MONITOR_CAPTURE_TIME"]) # this duration is added to the time it takes to capture and ML count
                    self.lazy_mode_frame_wait = float(conf["LAZY_MODE_FRAME_WAIT"])
                    self.focus_mode_frame_wait = float(conf["FOCUS_MODE_FRAME_WAIT"])
                    self.science_mode_time = float(conf["SCIENCE_MODE_TIME"])
                    self.persisted_write_time = float(conf["PERSISTED_WRITE_TIME"]) # every five minutes
                    self.loop_rate = float(conf["LOOP_RATE"]) # Rate float in Hz 
                    # close the file
                    process_file.close()

        self.state: int = -1
        self.mode: int = VisionProcessorMode.LAZY.value
        self.science_mode: bool = False
        self.science_mode_time: int = 0
        self.trigger_on: bool = False

        # a variable to emmit a thumbnail
        self.do_thumbnail: bool = False
        # the previous recieved settings
        self.previous_settings: bytes = None
        # the currently recieved settings
        self.current_settings: bytes = None
        # can persist settings
        self.can_persist = os.path.exists(self.options.persist_path)
        # if we have a valid file name for persisting camera settings
        if self.can_persist:
            # open the persisted configuration (in binary)
            with open(self.options.persist_path, 'rb') as persist_file:
                # get the persisted setting
                self.current_settings = persist_file.read()
                # close the file
                persist_file.close()

        self.loaded_model: Optional[LoadedModel] = None

         # the previous received settings
        self.previous_model_msg: bytes = None

        # the currently received settings
        self.current_model_msg: bytes = None

        self.image_source: ImageSource = self.setup_image_source(self.options, 640)

        # publish topics
        self.topic_image_stream: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_IMAGE_STREAM)
        self.topic_results: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_RESULTS)
        self.topic_state: str = comms.get_topic_for_camera(self.identifier, comms.TOPIC_POSTFIX_STATE)
        self.topic_ip_address: str = comms.get_topic_for_camera(self.identifier, comms.lite.TOPIC_POSTFIX_IP_ADDRESS)

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
        self.logger.debug(f'{message.topic}: {message.payload}')
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
            is_monitoring: bool = self.mode == VisionProcessorMode.MONITORING.value
            # is in science mode
            self.science_mode = is_monitoring and comms.unpack_bool_message(message.payload)
            # if in science mode
            if self.science_mode:
                # get current moment for timeout
                self.science_mode_time = time.time()
            else:
                self.science_mode_time = time.time() - self.science_mode_time

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
        if self.loaded_model is not None and self.loaded_model.name == message.name:
            self.loaded_model.confidence_threshold = message.confidence_threshold
            self.loaded_model.iou = message.iou

            return

        to_load: Optional[Path] = None

        for model_file in self.options.models_path.glob('*.pt'):
            if model_file.stem != message.name:
                continue

            to_load = model_file
            break

        if to_load is None:
            self.logger.error(f'Unable to load model with name "{message.name}": File not found!')

            return

        model = YOLO(to_load)
        
        self.loaded_model = LoadedModel(message.name, model, message.confidence_threshold, message.iou)

        # TODO: Currently, updating the PiCamera2 configuration causes the image to be completely white...
        #       This could be due to the configuration issues discovered in the first deployment

        # Update the output length of the image source
        # self.image_source.update_output_length(self.model_size)

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
            return ImageSourcePiCam(model_size, self.process_image_neural, self.publish_thumbnail, self.logger, str(self.options.config_path))
    
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
            self.client.publish(self.topic_image_stream, frame)
            
        # restore the do thumbnail state
        self.do_thumbnail = False

    def process_image_neural(self, frame: numpy.ndarray) -> None:
        if len(frame) == 0:
            raise Exception('Camera produced a zero length frame!')

        if self.loaded_model is None:
            self.logger.warning('No model has been requested! Aborting processing...')

            return

        # encode the frame as JPEG
        frame_encoded: bytes = cv2.imencode('.jpeg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))[1].tobytes()
        
        # if in science mode
        if self.science_mode:
            # get the frame shape
            height, width, _ = frame.shape
            # get ration
            camera_ratio: float = height / width
            # the image size used for raw images in ML
            output_height: int = self.loaded_model.size
            output_width: int = self.loaded_model.size
            # scale the correct dimension
            if width > height:
                output_height = int(round(self.loaded_model.size * camera_ratio))
            elif height > width:
                output_width = int(round(self.loaded_model.size / camera_ratio))
            # print("ML frame ", output_width, output_height)
            # create the resized frame
            frame_model_sized = cv2.resize(frame, dsize=(output_width, output_height), interpolation=CAPTURE_DOWNSAMPLE_METHOD)
        else:
            # publish the bytes
            # the frame is already the right size
            frame_model_sized = frame

        # set the processing state
        self.update_state(VisionProcessorState.PROCESSING)
        # Set the model with the raw frame
        results: Results = self.loaded_model.process(frame_model_sized, agnostic_nms=True, max_det=999)
        label_count: int = len(self.loaded_model.model.names)
        result_count: int = len(results)

        self.logger.info(f'Detected {result_count} corals!')

        boxes: List[comms.Box] = []

        for i in range(result_count):
            label = int(results.boxes.cls[i].item())
            boxes.append(comms.Box(*results.boxes.xyxyn[i], label=label))
        
        # include volume calc in litres
        sampled_volume: float = self.image_source.get_dof_volume() * 1e-6
        self.logger.debug(f'Volume litres: {sampled_volume}')
        
        self.client.publish(self.topic_image_stream, frame_encoded)
        self.client.publish(self.topic_results, comms.ResultMessage(frame_encoded, sampled_volume, label_count, boxes).pack())

    def setup_mqtt(self, options: CslicsArgs) -> None:
        self.logger.info(f'Connecting to MQTT broker at {options.broker_host}:{options.broker_port}...')
        # define a connection test variable
        connected: bool = False
        # while the program is running and not connected to the MQTT broker
        while self.is_running and not connected:
            # try to connect
            try:
                self.client.connect(options.broker_host, options.broker_port)
                connected = True
            except:
                time.sleep(1.0)
        # if conencted
        if connected:
            self.logger.info('Connected to MQTT broker!')
            self.client.loop_start()

            # publish the identifier
            self.publish_identifier()

            # publish the IP address
            ip_address: str = get_ip_address()
            self.logger.info(f'IP Address of this camera: {ip_address}')
            self.client.publish(self.topic_ip_address, ip_address, retain=True)

    
    def loop(self) -> None:
        # get the current time in seconds
        t0_id = time.time()
        t0_mon = t0_id
        t0_persist = t0_id
        # set the loop rate as a wait time
        loop_rate = 1.0/self.loop_rate
        # the program loop
        while self.is_running: 
            # sleep for 1/rate seconds
            time.sleep(loop_rate)
            # get the current time running
            t1 = time.time()
            # if time to publish a heartbeat
            if (t1 - t0_id) >= self.heartbeat_rate:
                # publish the device identifier
                self.publish_identifier()
                # restart the stop watch
                t0_id = t1
            # if time to write current setting
            if self.can_persist and (t1 - t0_persist) >= self.persisted_write_time:
                # update persist timer
                t0_persist = t1
                # open the persisted configuration (in binary)
                with open(self.options.persist_path, 'wb') as persist_file:
                    print("Writing to persist file")
                    # write the persisted setting
                    persist_file.write(self.current_settings) 
                    # close file
                    persist_file.close()

            # doing a science mode publish
            if self.science_mode:
                # if timed out
                if (t1 - self.science_mode_time) >= self.science_mode_time:
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
                except Exception as e:
                    self.logger.exception(e)
                # reset the state change
                self.previous_model_msg = self.current_model_msg
            # define the Modes
            if self.mode == VisionProcessorMode.LAZY.value:
                # start the camera thumbnail stream
                self.image_source.start()
                # if time to publish a thumbnail
                if (t1 - t0_mon) >= self.lazy_mode_frame_wait:
                    # update timer
                    t0_mon = t1
                    # trigger a do thumbnail
                    self.do_thumbnail = True
            elif self.mode == VisionProcessorMode.FOCUS_ADJUST.value:
                # start the camera thumbnail stream
                self.image_source.start()
                # if time to publish a thumbnail
                if (t1 - t0_mon) >= self.focus_mode_frame_wait:
                    # update timer
                    t0_mon = t1
                    # trigger a do thumbnail
                    self.do_thumbnail = True
            elif self.mode == VisionProcessorMode.MONITORING.value: 
                # if the capture has been triggered
                if self.trigger_on:
                    # test for state transitions
                    if self.state == VisionProcessorState.IDLE and (t1 - t0_mon) >= self.monitor_idle_time:
                        # update timer
                        t0_mon = t1
                        # update the state
                        self.update_state(VisionProcessorState.PRE_IMAGING)
                    elif self.state == VisionProcessorState.PRE_IMAGING and (t1 - t0_mon) >= self.monitor_pre_time:
                        # update timer
                        t0_mon = t1
                        # update the state
                        self.update_state(VisionProcessorState.IMAGING)
                    elif self.state == VisionProcessorState.IMAGING and (t1 - t0_mon) >= self.monitor_capture_time:
                        # capture the image and perfrom ML count
                        self.image_source.capture(int(self.science_mode == True))
                        # return to Idle state
                        self.update_state(VisionProcessorState.IDLE)
                        # update timer
                        t0_mon = t1
                        # restore the trigger state
                        self.trigger_on = False
            # make sure we are still connected
            if not self.client.is_connected():
                 # try to reconnect
                 self.setup_mqtt(self.options)


def main() -> None:
    logging.basicConfig()
    logger: Logger = logging.getLogger(SOFTWARE_NAME)
    logger.setLevel(logging.DEBUG)

    logger.info(f'Starting {SOFTWARE_TAG}')

    options = CslicsArgs()
    
    client = CslicsClient(options, logger)
    client.loop()
    client.image_source.close()

if __name__ == '__main__':
    main()
