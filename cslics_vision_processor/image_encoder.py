#!/usr/bin/env python3

# Public licence for commercial/non-commercial use RRA (internationally)
# 
# Each Project IP Owner grants to, or must obtain for, each other Party and any member of the public a perpetual, irrevocable, worldwide, non-exclusive, royalty-free, non-transferable licence (including a right of sub-license to any person (in the case of GBRF including, but not limited to, the Department)) to Use the Project IP and Project Improvements, in the field of reef restoration and adaptation, for:
# 
#   (a) non-commercial purposes, educational and/or research purposes (including for the performance of Core Commonwealth Functions by the Department); and/or
#   (b) commercial purposes, whether in Australia or elsewhere.
# 
# Each Project IP Owner acknowledges that any licence granted to the Department for Core Commonwealth Functions will not be restricted to use in the field of reef restoration and adaptation.
# 
# Derivative works must be distributed with a copy of this licence which does not further restrict the rights of licensees.
# 
# Amendment providing additional Limitation of Liability for QUT
# 
# QUT does not warrant that:
# 
#   (a) software/code is fit for the Approved Purpose, or that it has any particular qualities or characteristics;
#   (b) the software/code is free from errors, viruses, worms, or similar defects;
#   (c) the use of the software/code by the Licensee will lead to any particular result; or
#   (d) the use of the software/code will not infringe the rights (including Intellectual Property rights) of any person.
# 
# Copyright (C) 2025 Queensland University of Technology
# 

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
