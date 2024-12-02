
# Common unit conversion factors used in aerospace engineering
length_conversion_factors = {
    'meters_to_feet': 3.28084,
    'feet_to_meters': 0.3048,
    'kilometers_to_miles': 0.621371,
    'miles_to_kilometers': 1.60934,
    'meters_to_inches': 39.3701,
    'inches_to_meters': 0.0254,
}

pressure_conversion_factors = {
    'pascals_to_psi': 0.000145038,
    'psi_to_pascals': 6894.76,
    'pascals_to_inches_of_mercury': 0.0002953,
    'inches_of_mercury_to_pascals': 3386.39,
    'pascals_to_millibars': 0.01,
    'millibars_to_pascals': 100,
    'pascals_to_atmospheres': 0.00000986923,
    'atmospheres_to_pascals': 101325,
}

force_conversion_factors = {
    'newtons_to_pounds': 0.224809,
    'pounds_to_newtons': 4.44822,
    'kilograms_to_pounds': 2.20462,
    'pounds_to_kilograms': 0.453592,
    'dynes_to_pounds': 0.00001,
    'pounds_to_dynes': 100000,
}

mass_conversion_factors = {
    'kilograms_to_slugs': 0.0685218,
    'slugs_to_kilograms': 14.5939,
    'kilograms_to_pounds': 2.20462,
    'pounds_to_kilograms': 0.453592,
    'grams_to_ounces': 0.035274,
    'ounces_to_grams': 28.3495,
}

volume_conversion_factors = {
    'liters_to_gallons': 0.264172,
    'gallons_to_liters': 3.78541,
}

temperature_conversion_factors = {
    'degrees_celsius_to_degrees_fahrenheit': lambda c: (c * 9/5) + 32,
    'degrees_fahrenheit_to_degrees_celsius': lambda f: (f - 32) * 5/9,
}

angle_conversion_factors = {
    'radians_to_degrees': lambda r: r * (180 / 3.14159),
    'degrees_to_radians': lambda d: d * (3.14159 / 180),
}

energy_conversion_factors = {
    'joules_to_foot_pounds': 0.737562,
    'foot_pounds_to_joules': 1.35582,
}

# Combine all conversion factors into a single dictionary for easy access
unit_conversion_factors = {
    'length': length_conversion_factors,
    'pressure': pressure_conversion_factors,
    'force': force_conversion_factors,
    'mass': mass_conversion_factors,
    'volume': volume_conversion_factors,
    'temperature': temperature_conversion_factors,
    'angle': angle_conversion_factors,
    'energy': energy_conversion_factors,
}
