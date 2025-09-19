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
