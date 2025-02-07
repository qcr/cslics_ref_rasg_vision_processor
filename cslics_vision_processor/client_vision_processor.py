#!/usr/bin/env python3

import numpy, logging, json, time, cv2, socket
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
SOFTWARE_VERSION: str = 'v1.8'
SOFTWARE_TAG: str = f'{SOFTWARE_NAME} {SOFTWARE_VERSION}'

#: the method being used to down-sample the raw frame image for ML
CAPTURE_DOWNSAMPLE_METHOD = cv2.INTER_AREA


def get_ip_address() -> str:
    """ Determine the IP address likely used by this device on the local network.

    Returns:
        The IP address as a `str`. The loopback address will be returned in the case that no address could be found.
    """
    
    try:
        return [(s.connect(('255.255.0.0', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]
    except:
        return '127.0.0.1'


class ImageSourceType(Enum):
    """Enumerating the supported image sources."""

    PICAM = 0
    STORAGE_LOCAL = 1


class CslicsArgs:
    """A class for handling parameters for the `CslicsClient`."""

    def __init__(self):
        self.__broker_host: str = 'localhost'
        self.__broker_port: int = 1883
        self.__image_source: ImageSourceType = ImageSourceType.PICAM
        self.__image_directory: Optional[Path] = None
        self.__process_path: Optional[Path] = None
        self.__identifier: Optional[str] = None

        parser = ArgumentParser(SOFTWARE_TAG, description='CSLICS client for edge computing devices')
        parser.add_argument('-b', '--broker-host', metavar='host', default=self.__broker_host, help='URI for the MQTT broker host')
        parser.add_argument('-p', '--broker-port', metavar='port', default=self.__broker_port, type=int, help='Port for the MQTT broker host')
        parser.add_argument('-m', '--models-path', required=True, metavar='/path/to/model/directory/', help='Path to the model files to be used on the CSLICS Vision Processor')
        parser.add_argument('--image-source', required=False, default=self.__image_source, choices=ImageSourceType.__members__, help='Source to use for acquiring images')
        arg_image_directory = parser.add_argument('--image-directory', required=False, help=f'Directory to use for images if {ImageSourceType.STORAGE_LOCAL.name} is the selected image source')
        parser.add_argument('-id', '--identifier', required=False, help=f'Override the identifier discovery')
        parser.add_argument('--config_file_path', required=True, help=f'The file path to the camera configuration JSON file')
        parser.add_argument('--process_conf_path', required=False, default=None, help=f'The file path to the camera process configuration JSON file')

        args = parser.parse_args()

        self.__broker_host = args.broker_host
        self.__broker_port = args.broker_port
        self.__models_path = Path(args.models_path)
        self.__image_source = ImageSourceType[args.image_source]
        self.__image_directory = args.image_directory
        self.__identifier = args.identifier
        self.__config_path = Path(args.config_file_path)
        self.__process_path = Path(args.process_conf_path) if args.process_conf_path is not None else None

        if self.image_source is ImageSourceType.STORAGE_LOCAL and self.image_directory == None:
            raise ArgumentError(arg_image_directory, f'Argument must be specified when using image_source {ImageSourceType.STORAGE_LOCAL.name}!')

    @property
    def broker_host(self) -> str:
        """The IP address or URI of the MQTT broker host."""

        return self.__broker_host

    @property
    def broker_port(self) -> int:
        """The port of the MQTT broker host."""

        return self.__broker_port

    @property
    def models_path(self) -> Path:
        """The path to the storage location of the vision models."""

        return self.__models_path

    @property
    def image_source(self) -> ImageSourceType:
        """The image source to use when collecting image samples to process."""

        return self.__image_source

    @property
    def image_directory(self) -> Optional[str]:
        """The directory used to fetch images from when using the `ImageSourceType.STORAGE_LOCAL` image source."""
        
        return self.__image_directory

    @property
    def config_path(self) -> Path:
        """The path to the PiCamera2 configuration for the `ImageSourceType.PICAM` image source."""

        return self.__config_path

    @property
    def process_path(self) -> Optional[Path]:
        """The path to the configuration for timing adjustments."""

        return self.__process_path
    
    @property
    def identifier(self) -> Optional[str]:
        """An optional override for the UUID of the `CslicsClient`."""

        return self.__identifier
    
    # A setter is included for this property as it is required for simulating many cameras.
    @identifier.setter
    def identifer(self, value: str) -> None:
        self.__identifier = value


class LoadedModel:
    """A class to handle the configuration and running of loaded vision models."""

    __name: str
    __model: YOLO
    __size: int
    __is_fused: bool = False

    #: The "conf" parameter to use when running the model.
    confidence_threshold: float

    #: The "iou" parameter to use when running the model.
    iou: float

    def __init__(self, name: str, model: YOLO, confidence_threshold: float = 0.7, iou: float = 0.5):
        """
        Args:
            name: The name of the model. Typically the "stem" of the file.
            model: The loaded `YOLO` model.
            confidence_threshold: The "conf" parameter to use when running the model.
            iou: The "iou" parameter to use when running the model.
        """

        self.__name = name
        self.__model = model
        self.__size = self.__model.overrides['imgsz']
        self.confidence_threshold = confidence_threshold
        self.iou = iou

    @property
    def name(self) -> str:
        """The name of the model. Typically the "stem" of the file."""
        
        return self.__name
        
    @property
    def model(self) -> YOLO:
        """The loaded `YOLO` model."""

        return self.__model

    @property
    def size(self) -> int:
        """The image size used by the model."""

        return self.__size

    @property
    def is_fused(self) -> bool:
        """Whether this model has had `fuse` invoked upon it."""

        return self.__is_fused

    def fuse(self) -> None:
        """Fuse the YOLO model."""

        self.__model.fuse()
        self.__is_fused = True

    def process(self, source: numpy.ndarray, **model_kwargs) -> Results:
        """Process an image with the model and return the results.
        
        Args:
            source: The image to process with the model.
            model_kwargs: Additional kwargs to pass through to the model.
        """

        if not self.is_fused:
            self.fuse()

        return self.__model(source, conf=self.confidence_threshold, iou=self.iou, **model_kwargs)[0]


class CslicsClient:
    """The client handler for a CSLICS Vision Processor."""

    def __init__(self, options: CslicsArgs, logger: Logger):
        """
        Args:
            options: The arguments for configuring this client.
            logger: The logger to create a child from for this object.
        """

        self.__logger: Logger = logger.getChild(CslicsClient.__name__)
        self.__is_running: bool = True
        self.__options: CslicsArgs = options
        self.__identifier: str = self.__get_unique_identifier()

        # Timing configuration parameters
        self.__heartbeat_rate: float = 5.0
        self.__monitor_pre_time: float = 2.0
        self.__lazy_mode_frame_wait: float = 10.0
        self.__focus_mode_frame_wait: float = 0.2
        self.__science_mode_timeout: float = 1800.0
        self.__loop_rate = 20.0 # Rate float in Hz 

        # if given an existing path, otherwise just use default
        if self.__options.process_path is not None:
            if not self.__options.process_path.exists():
                self.__logger.warning('Process configuration path specified but does not exist!')
            else:
                with open(self.__options.process_path, 'r') as process_file:
                    # load the JSON
                    conf = json.load(process_file)
                    self.__heartbeat_rate = float(conf["HEARTBEAT_RATE"])
                    self.__monitor_pre_time = float(conf["MONITOR_PRE_TIME"])
                    self.__lazy_mode_frame_wait = float(conf["LAZY_MODE_FRAME_WAIT"])
                    self.__focus_mode_frame_wait = float(conf["FOCUS_MODE_FRAME_WAIT"])
                    self.__science_mode_timeout = float(conf["SCIENCE_MODE_TIME"])
                    self.__loop_rate = float(conf["LOOP_RATE"]) # Rate float in Hz 
                    # close the file
                    process_file.close()

        self.__state: int = -1
        self.__mode: int = VisionProcessorMode.LAZY.value
        self.__science_mode: bool = False
        self.__science_mode_request_time: float = 0.0
        self.__trigger_on: bool = False

        # a variable to emmit a thumbnail
        self.__do_thumbnail: bool = False
        # the previous recieved settings
        self.__previous_settings: bytes = None
        # the currently recieved settings
        self.__current_settings: bytes = None

        self.__loaded_model: Optional[LoadedModel] = None

         # the previous received settings
        self.__previous_model_msg: bytes = None

        # the currently received settings
        self.__current_model_msg: bytes = None

        self.__image_source: ImageSource = self.__setup_image_source(640)

        # publish topics
        self.__topic_image_stream: str = comms.get_topic_for_camera(self.__identifier, comms.TOPIC_POSTFIX_IMAGE_STREAM)
        self.__topic_results: str = comms.get_topic_for_camera(self.__identifier, comms.TOPIC_POSTFIX_RESULTS)
        self.__topic_state: str = comms.get_topic_for_camera(self.__identifier, comms.TOPIC_POSTFIX_STATE)
        self.__topic_ip_address: str = comms.get_topic_for_camera(self.__identifier, comms.TOPIC_POSTFIX_IP_ADDRESS)
        self.__topic_version: str = comms.get_topic_for_camera(self.__identifier, comms.TOPIC_POSTFIX_VERSION)

        #subscribe topics
        self.__topic_trigger: str = comms.get_topic_for_camera(self.__identifier, comms.TOPIC_POSTFIX_TRIGGER)
        self.__topic_settings: str = comms.get_topic_for_camera(self.__identifier, comms.TOPIC_POSTFIX_CAMERA_CONFIG)
        self.__topic_mode: str = comms.get_topic_for_camera(self.__identifier, comms.TOPIC_POSTFIX_MODE)
        self.__topic_model: str = comms.get_topic_for_camera(self.__identifier, comms.TOPIC_POSTFIX_MODEL)
        self.__topic_science: str = comms.get_topic_for_camera(self.__identifier, comms.TOPIC_POSTFIX_SCIENCE)

        # TODO: This is not robust to the MQTT broker disconnecting
        self.__client = Client(CallbackAPIVersion.VERSION2, f'{SOFTWARE_NAME}.{self.__identifier}')

        # set the MQTT message callback functions
        self.__client.on_message = self.__on_message
        self.__client.on_connect = self.__on_connect
        # initialise the MQTT client
        self.__setup_mqtt()

    def __on_connect(self, client, userdata, flags, reason_code, properties):
        """The MQTT client callback function that gets called when the MQTT client successfully.

        This method sets up the MQTT subscription topics.

        Args:
            client: Unused.
            userdata: Unused.
            flags: Unused.
            reason_code: Unused.
            properties: Unused.
        """

        # set the initial camera state to Idle
        self.__update_state(VisionProcessorState.IDLE)
        # setup subscriptions: for camera triggers
        self.__client.subscribe(self.__topic_trigger)
        # setup subscriptions: for camera configuration
        self.__client.subscribe(self.__topic_settings)
        # setup subscriptions: for mode of camera operations
        self.__client.subscribe(self.__topic_mode)
        # setup subscription for the model
        self.__client.subscribe(self.__topic_model)
        # setup subscriptions: for science mode of camera operations
        self.__client.subscribe(self.__topic_science)
    
    def __on_message(self, client: Client, userdata, message: MQTTMessage):
        """The MQTT subscribed message callback function.

        Caution! This callback is not invoked on the main thread!

        Args:
            client: Unused. The client instance for this callback.
            userdata: Unused. The private user data as set in Client() or user_data_set()
            message: The received message. This is a class with members: topic, payload, qos, retain.
        """

        self.__logger.debug(f'{message.topic}: {message.payload}')
        # Select the topic action
        if message.topic == self.__topic_trigger:
            # if in monitoring mode
            if self.__mode == VisionProcessorMode.MONITORING.value and not self.__trigger_on:
                self.__trigger_on = True
                self.__update_state(VisionProcessorState.IDLE)
        elif message.topic == self.__topic_settings:
            # set the current setting string
            self.__current_settings = message.payload
        elif message.topic == self.__topic_mode:
            # get the message as a string
            msg = str(message.payload, "utf-8")
            # check the message is digit
            if msg.isdigit():
                # get the mode
                the_mode = int(msg)
                # if we are already in this mode
                if  self.__mode == the_mode:
                    return
                # make sure the mode we received is within range
                if VisionProcessorMode.LAZY.value <= the_mode <= VisionProcessorMode.FOCUS_ADJUST.value:
                    # set the mode
                    self.__mode = the_mode
                    # initial the state
                    if self.__mode == VisionProcessorMode.LAZY.value:
                        # make sure the camera is stopped
                        self.__image_source.stop()
                        self.__update_state(VisionProcessorState.IDLE)
                    elif self.__mode == VisionProcessorMode.MONITORING.value:
                        # make sure the camera is stopped
                        self.__image_source.stop()
                        # set the state
                        self.__update_state(VisionProcessorState.IDLE)
                    elif self.__mode == VisionProcessorMode.FOCUS_ADJUST.value:
                        # make sure the camera is stopped
                        self.__image_source.stop()
                        # publish once the focus adjust state
                        self.__update_state(VisionProcessorState.FOCUS_ADJUST)
        elif message.topic == self.__topic_model:
            # set the model ,message
            self.__current_model_msg = message.payload   
        elif message.topic == self.__topic_science:
            # is in science mode
            self.__science_mode = comms.unpack_bool_message(message.payload)
            # if in science mode
            if self.__science_mode:
                # get current moment for timeout
                self.__science_mode_request_time = time.time()
            else:
                self.__science_mode_request_time = time.time() - self.__science_mode_timeout

    def __get_unique_identifier(self) -> str:
        """Determine the unique identifier to use for communications on the MQTT network.

        Returns:
            A unique identifer for representing this client.
        """

        if self.__options.identifier is not None:
            self.__logger.warning(f'Using override identifier: {self.__options.identifer}')
            return self.__options.identifer
        
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if not line.startswith('Serial'):
                        continue

                    return line.split(':')[1].strip()
        except:
            pass

        self.__logger.warning('Unable to find a source of unique ID... Generating one!')

        # In the case we cannot find one, a random one will do
        import random
        return ''.join(random.choice('0123456789ABCDEF') for i in range(16))
    
    def __update_model(self, message: comms.ModelMessage) -> None:
        """ Given a model message, updates the loaded YOLO model and associated parameters.

        Args:
            message: The model message received from the MQTT broker.
        """

        if self.__loaded_model is not None and self.__loaded_model.name == message.name:
            self.__loaded_model.confidence_threshold = message.confidence_threshold
            self.__loaded_model.iou = message.iou

            return

        to_load: Optional[Path] = None

        for model_file in self.__options.models_path.glob('*.pt'):
            if model_file.stem != message.name:
                continue

            to_load = model_file
            break

        if to_load is None:
            self.__logger.error(f'Unable to load model with name "{message.name}": File not found!')

            return

        model = YOLO(to_load)
        
        self.__loaded_model = LoadedModel(message.name, model, message.confidence_threshold, message.iou)

        # Update the output length of the image source
        self.__image_source.update_output_length(self.__loaded_model.size)

    def __update_state(self, state: VisionProcessorState) -> None:
        """Update the state of this client and publish that state to the MQTT network.

        Args:
            state: The state to update the client to.
        """

        self.__state = state
        self.__client.publish(self.__topic_state, state.value, retain=True)

    def __setup_image_source(self, default_image_size: int) -> ImageSource:
        """Set up the image source as requested by `CslicsArgs.image_source`.

        Args:
            default_image_size: The default image size to request the `ImageSource` to produce.

        Returns:
            The instantiated image source.
        """

        self.__logger.info('Setting up image source...')
        
        if self.__options.image_source == ImageSourceType.STORAGE_LOCAL:
            from cslics_vision_processor.imaging import ImageSourceStorageLocal
            return ImageSourceStorageLocal(default_image_size, self.__process_image_neural, self.__publish_thumbnail, self.__options.image_directory, self.__logger)
        
        if self.__options.image_source == ImageSourceType.PICAM:
            from cslics_vision_processor.imaging import ImageSourcePiCam
            return ImageSourcePiCam(default_image_size, self.__process_image_neural, self.__publish_thumbnail, self.__logger, self.__options.config_path)
    
    def __publish_identifier(self) -> None:
        """Publish this clients UUID to the MQTT network."""
        self.__client.publish(comms.TOPIC_CAMERAS, self.__identifier)

    def __publish_thumbnail(self, frame: bytes) -> None:
        """Callback for when a compressed image is produced by the image source.

        This image should (under most circumstances) be published to the MQTT network for view by users.

        Caution: This callback may be on another thread!
        
        Args:
            frame: The compressed image produced by the image source.
        """

        # if not doing a thumbnail
        if not self.__do_thumbnail:
            return
        if self.__mode == VisionProcessorMode.LAZY.value or self.__mode == VisionProcessorMode.FOCUS_ADJUST.value:
            self.__logger.info(f'Live-view length (bytes): {len(frame)}. Publishing...')
            self.__client.publish(self.__topic_image_stream, frame)
            
        # restore the do thumbnail state
        self.__do_thumbnail = False

    def __process_image_neural(self, frame: numpy.ndarray) -> None:
        """Callback for when an uncompressed image is produced by the image source.

        Handle science mode communications.
        Process the image using the loaded vision model.
        Publish the results over the MQTT network.

        Args:
            frame: The raw frame produced by the image source.
        """

        if len(frame) == 0:
            raise Exception('Camera produced a zero length frame!')

        if self.__loaded_model is None:
            self.__logger.warning('No model has been requested! Aborting processing...')

            return

        # encode the frame as JPEG
        frame_encoded: bytes = cv2.imencode('.jpeg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))[1].tobytes()
        
        # if in science mode
        if self.__science_mode:
            # get the frame shape
            height, width, _ = frame.shape
            # get ration
            camera_ratio: float = height / width
            # the image size used for raw images in ML
            output_height: int = self.__loaded_model.size
            output_width: int = self.__loaded_model.size
            # scale the correct dimension
            if width > height:
                output_height = int(round(self.__loaded_model.size * camera_ratio))
            elif height > width:
                output_width = int(round(self.__loaded_model.size / camera_ratio))
            # print("ML frame ", output_width, output_height)
            # create the resized frame
            frame_model_sized = cv2.resize(frame, dsize=(output_width, output_height), interpolation=CAPTURE_DOWNSAMPLE_METHOD)
        else:
            # publish the bytes
            # the frame is already the right size
            frame_model_sized = frame

        # set the processing state
        self.__update_state(VisionProcessorState.PROCESSING)
        # Set the model with the raw frame
        results: Results = self.__loaded_model.process(frame_model_sized, agnostic_nms=True, max_det=999)
        label_count: int = len(self.__loaded_model.model.names)
        result_count: int = len(results)

        self.__logger.info(f'Detected {result_count} corals!')

        boxes: List[comms.Box] = []

        for i in range(result_count):
            label = int(results.boxes.cls[i].item())
            boxes.append(comms.Box(*results.boxes.xyxyn[i], label=label))
        
        # include volume calc in litres
        sampled_volume: float = self.__image_source.get_dof_volume() * 1e-3
        self.__logger.debug(f'Volume litres: {sampled_volume}')
        
        self.__client.publish(self.__topic_image_stream, frame_encoded)
        self.__client.publish(self.__topic_results, comms.ResultMessage(frame_encoded, sampled_volume, label_count, boxes).pack())

    def __setup_mqtt(self) -> None:
        """Establish the connection to the MQTT broker and publish metadata about this client."""

        self.__logger.info(f'Connecting to MQTT broker at {self.__options.broker_host}:{self.__options.broker_port}...')
        # define a connection test variable
        connected: bool = False
        # while the program is running and not connected to the MQTT broker
        while self.__is_running and not connected:
            # try to connect
            try:
                self.__client.connect(self.__options.broker_host, self.__options.broker_port)
                connected = True
            except:
                time.sleep(1.0)
        # if conencted
        if connected:
            self.__logger.info('Connected to MQTT broker!')
            self.__client.loop_start()

            # publish the identifier
            self.__publish_identifier()

            # publish the IP address
            ip_address: str = get_ip_address()
            self.__logger.info(f'IP Address of this camera: {ip_address}')
            self.__client.publish(self.__topic_ip_address, ip_address, retain=True)
            self.__client.publish(self.__topic_version, SOFTWARE_VERSION, retain=True)
    
    def loop(self) -> None:
        """The main loop for handling CSLICS Vision Processor client operations."""

        # get the current time in seconds
        t0_id = time.time()
        t0_mon = t0_id
        # set the loop rate as a wait time
        loop_rate = 1.0/self.__loop_rate
        # the program loop
        while self.__is_running: 
            # sleep for 1/rate seconds
            time.sleep(loop_rate)
            # get the current time running
            t1 = time.time()
            # if time to publish a heartbeat
            if (t1 - t0_id) >= self.__heartbeat_rate:
                # publish the device identifier
                self.__publish_identifier()
                # restart the stop watch
                t0_id = t1

            # doing a science mode publish
            if self.__science_mode:
                # if timed out
                if (t1 - self.__science_mode_request_time) >= self.__science_mode_timeout:
                    self.__science_mode = False
            if self.__previous_settings != self.__current_settings:
                #try:
                # parse the settings
                settings: comms.CameraSettings = comms.CameraSettings.from_buffer(self.__current_settings)
                # start the camera thumbnail stream
                self.__image_source.start()
                # set camera
                self.__image_source.set_settings(settings)
                #except:
                #    self.logger.error("comms.CameraSettings could not parse the camera settings message.")
                # reset change
                self.__previous_settings = self.__current_settings
            if self.__previous_model_msg != self.__current_model_msg:
                try:
                    # parse the model message
                    msg: comms.ModelMessage = comms.ModelMessage.from_buffer(self.__current_model_msg)
                    # update the YOLO model
                    self.__update_model(msg)
                except Exception as e:
                    self.__logger.exception(e)
                # reset the state change
                self.__previous_model_msg = self.__current_model_msg
            # define the Modes
            if self.__mode == VisionProcessorMode.LAZY.value:
                # start the camera thumbnail stream
                self.__image_source.start()
                # if time to publish a thumbnail
                if (t1 - t0_mon) >= self.__lazy_mode_frame_wait:
                    # update timer
                    t0_mon = t1
                    # trigger a do thumbnail
                    self.__do_thumbnail = True
            elif self.__mode == VisionProcessorMode.FOCUS_ADJUST.value:
                # start the camera thumbnail stream
                self.__image_source.start()
                # if time to publish a thumbnail
                if (t1 - t0_mon) >= self.__focus_mode_frame_wait:
                    # update timer
                    t0_mon = t1
                    # trigger a do thumbnail
                    self.__do_thumbnail = True
            elif self.__mode == VisionProcessorMode.MONITORING.value: 
                # if the capture has been triggered
                if self.__trigger_on:
                    # test for state transitions
                    if self.__state == VisionProcessorState.IDLE:
                        # update timer
                        t0_mon = t1
                        # update the state
                        self.__update_state(VisionProcessorState.PRE_IMAGING)
                    elif self.__state == VisionProcessorState.PRE_IMAGING and (t1 - t0_mon) >= self.__monitor_pre_time:
                        # update timer
                        t0_mon = t1
                        # update the state
                        self.__update_state(VisionProcessorState.IMAGING)
                        # capture the image and perfrom ML count
                        self.__image_source.capture(int(self.__science_mode == True))
                        # return to Idle state
                        self.__update_state(VisionProcessorState.IDLE)
                        # update timer
                        t0_mon = t1
                        # restore the trigger state
                        self.__trigger_on = False
            # make sure we are still connected
            if not self.__client.is_connected():
                 # try to reconnect
                 self.__setup_mqtt()

    def shutdown(self) -> None:
        """Shut down this client."""

        self.__is_running = False
        self.__image_source.close()


def main() -> None:
    """Instantiate and run a CSLICS Client Vision Processor."""

    logging.basicConfig()
    logger: Logger = logging.getLogger(SOFTWARE_NAME)
    logger.setLevel(logging.DEBUG)

    logger.info(f'Starting {SOFTWARE_TAG}')

    options = CslicsArgs()
    
    client = CslicsClient(options, logger)
    client.loop()
    client.shutdown()

if __name__ == '__main__':
    main()
