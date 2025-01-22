#!/usr/bin/env python3

# Author:   Alec Tutin
# Date:     2024-11-18

from libcamera import controls

def as_AeConstraintModeEnum(name: str) -> controls.AeConstraintModeEnum:
    """A helper function to convert a `str` name to a value of `libcamera`'s `controls.AeConstraintModeEnum`.
    
    Args:
        name: A `str` representation of an `controls.AeConstraintModeEnum` value.
    """

    return getattr(controls.AeConstraintModeEnum, '__entries')[name][0]

def as_AeExposureModeEnum(name: str) -> controls.AeExposureModeEnum:
    """A helper function to convert a `str` name to a value of `libcamera`'s `controls.AeExposureModeEnum`.
    
    Args:
        name: A `str` representation of an `controls.AeExposureModeEnum` value.
    """

    return getattr(controls.AeExposureModeEnum, '__entries')[name][0]

def as_AwbModeEnum(name: str) -> controls.AwbModeEnum:
    """A helper function to convert a `str` name to a value of `libcamera`'s `controls.AwbModeEnum`.
    
    Args:
        name: A `str` representation of an `controls.AwbModeEnum` value.
    """

    return getattr(controls.AwbModeEnum, '__entries')[name][0]
