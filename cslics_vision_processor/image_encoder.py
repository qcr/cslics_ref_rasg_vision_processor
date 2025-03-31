#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2025-03-31

import cv2
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
        if not self.__on_job_completed.wait(timeout):
            raise TimeoutError('Timed out waiting for image encoding job to complete!')
        
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
