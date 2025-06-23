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
