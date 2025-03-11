#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-07-30

import logging, random
from client_vision_processor import CslicsArgs, CslicsClient
from typing import List
from logging import Logger
from copy import copy
from threading import Thread

SOFTWARE_NAME: str = 'simulate_cameras'
SOFTWARE_VERSION: str = 'v0.0'
SOFTWARE_TAG: str = f'{SOFTWARE_NAME} {SOFTWARE_VERSION}'

def get_unique_identifier() -> str:
    return ''.join(random.choice('0123456789ABCDEF') for i in range(16))

def main() -> None:
    logging.basicConfig()
    logger: Logger = logging.getLogger(SOFTWARE_NAME)
    logger.setLevel(logging.DEBUG)

    logger.info(f'Starting {SOFTWARE_TAG}')

    random.seed(1337)

    options: CslicsArgs = CslicsArgs()
    
    clients: List[CslicsClient] = []
    threads: List[Thread] = []
    
    while True:
        try:
            input(f'{len(clients)} simulated vision processors running. Press Enter to start another and CTRL+D to quit!')
        except:
            print() #CTRL+D will not create a newline on the CLI
            break

        # In the case we cannot find one, a random one will do

        client_options: CslicsArgs = copy(options)
        client_options.identifer = get_unique_identifier()
        client: CslicsClient = CslicsClient(client_options, logger.getChild(client_options.identifer))
        clients.append(client)

        client_thread: Thread = Thread(target=client.loop)
        threads.append(client_thread)
        client_thread.start()
    
    print('Closing client threads...')

    for client in clients:
        client.shutdown()

    for thread in threads:
        thread.join()
    
    print('Done!')

if __name__ == '__main__':
    main()
