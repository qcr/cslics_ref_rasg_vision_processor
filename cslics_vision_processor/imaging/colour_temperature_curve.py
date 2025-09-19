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
# Date:     2025-03-10

from typing import List, Tuple

class ColourTemperatureCurve:
    def __init__(self, curve_definition: List[float]):
        self.__curve: List[float] = curve_definition

    def sample(self, value: float) -> Tuple[float, float]:
        """Given a normalised requested colour temperature, sample the curves to produce red and blue gains.
        
        Args:
            normalised: The normalised colour temperature request value.

        Returns:
            A tuple of the red and blue gains.
        """

        stride: int = 3
        temperature_min: float = self.__curve[0]
        temperature_max: float = self.__curve[-stride]
        temperature_requested: float = temperature_min + (temperature_max - temperature_min) * value

        for i in range(0, len(self.__curve) - stride, stride):
            curve_temperature: float = self.__curve[i]

            if curve_temperature < temperature_requested:
                continue

            temperature_curve_next: float = self.__curve[i + stride]
            lerp_t: float = (temperature_requested - curve_temperature) / (temperature_curve_next - curve_temperature)

            red_lower: float = self.__curve[i + 1]
            red_upper: float = self.__curve[i + 1 + stride]
            red_result: float = red_lower + (red_upper - red_lower) * lerp_t

            blue_lower: float = self.__curve[i + 2]
            blue_upper: float = self.__curve[i + 2 + stride]
            blue_result: float = blue_lower + (blue_upper - blue_lower) * lerp_t

            return (red_result, blue_result)

        return (self.__curve[-2], self.__curve[-1])
