#!/bin/bash

rm -rf docs/

pdoc3 --html cslics_mqtt cslics_vision_processor -c show_source_code=False -o ./docs
