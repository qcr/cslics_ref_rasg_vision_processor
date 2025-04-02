#!/usr/bin/env python3

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
