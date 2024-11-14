#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-11-18

from libcamera import controls

def as_AeConstraintModeEnum(name: str) -> controls.AeConstraintModeEnum:
    return getattr(controls.AeConstraintModeEnum, '__entries')[name][0]

def as_AeExposureModeEnum(name: str) -> controls.AeExposureModeEnum:
    return getattr(controls.AeExposureModeEnum, '__entries')[name][0]

def as_AwbModeEnum(name: str) -> controls.AwbModeEnum:
    return getattr(controls.AwbModeEnum, '__entries')[name][0]
