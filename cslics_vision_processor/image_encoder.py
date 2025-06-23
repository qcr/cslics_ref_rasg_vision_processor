#!/usr/bin/env python3

# Copyright 2024 Queensland University of Technology.
#
# The programming code herein is licensed to The Australian Institute of Marine Science (AIMS)
# by The Queensland University of Technology (QUT) to use for testing and validation of the
# Coral Spawn and Larvae Imaging Camera System (CSLICS).
# 
# All liabilities and guarantees for this program code and its supporting components are as stipulated
# in the relevant agreements relating to CSLICS between AIMS and QUT and by the licenses of the
# supporting components where made by a third party. QUT accepts no liability for modifications made
# to the programming code by parties other than QUT.

# Author:   Alec Tutin
# Date:     2025-03-31

import cv2
from cslics_vision_processor.imaging import ImageEncodingFailureException
from cv2.typing import MatLike
from threading import Event, Thread
from queue import Queue
from typing import Optional

class EncodeJob:
    """A class for representing an individual asynchronous encoding task."""

    def __init__(self, image: MatLike):
        self.image_raw: MatLike = image
        self.image_encoded: Optional[bytes] = None
        self.__on_job_completed = Event()

    def complete_job(self, image_encoded: bytes) -> None:
        self.image_encoded = image_encoded
        self.__on_job_completed.set()

    def error_job(self) -> None:
        self.__on_job_completed.set()

    def wait(self, timeout: Optional[float]) -> bytes:
        """Wait for the encoding job to be completed.
        
        Args:
            timeout: The length of time to wait in seconds before timing out. If `None`, will wait indefinitely.

        Raises:
            `ImageEncodingFailureException`: If the timeout was reached without the encoding being completed or if there was an error during encoding.
        """

        if not self.__on_job_completed.wait(timeout):
            raise TimeoutError('Timed out waiting for image encoding job to complete!')
        
        if self.image_encoded is None:
            ImageEncodingFailureException('Unable to encode captured image!')

        return self.image_encoded


class ImageEncoder:
    """A class which will perform asynchronous image encoding."""

    def __init__(self):
        self.__is_running: bool = True
        self.__queue: Queue[EncodeJob] = Queue()
        self.__thread = Thread(target=self.__loop, name=f'{ImageEncoder.__name__}')
        self.__thread.start()

    def __loop(self) -> None:
        while self.__is_running:
            job: EncodeJob = self.__queue.get()

            if job is None:
                self.__queue.task_done()
                continue

            success, encoded = cv2.imencode('.jpeg', job.image_raw)

            if not success:
                job.error_job()
            else:
                job.complete_job(encoded.tobytes())

            self.__queue.task_done()

    def encode(self, image: MatLike) -> EncodeJob:
        job = EncodeJob(image)
        self.__queue.put(job)

        return job

    def close(self):
        self.__is_running = False
        self.__queue.put(None)
