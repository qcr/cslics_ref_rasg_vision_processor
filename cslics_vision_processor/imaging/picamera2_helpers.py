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
