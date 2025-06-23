#!/bin/bash

# Copyright 2025 Queensland University of Technology.
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
# Date:     2025-06-23

cd $(dirname "$0")

package_path="cslics_vision_processor"
include_directories=("$package_path" "services" "config")
include_files=(LICENSE client_vision_processor.py)

# Relies on the use of single quotes for strings...
version=`cat ${package_path}/__init__.py | grep SOFTWARE_VERSION | cut -d"'" -f2`

archive_stem="${package_path}_${version}"
archive="$archive_stem.zip"

mkdir -p "build/$archive_stem"

for directory in ${include_directories[@]}; do
    echo "Copying ${directory}/ to build..."
    cp -R "$directory" "build/$archive_stem"
done

for file in ${include_files[@]}; do
    echo "Copying $file to build..."
    cp "$file" "build/$archive_stem"
done

cd build

find -name "__pycache__" | xargs rm -rf

if [ -f "$archive" ]; then
    rm "$archive"
fi

zip -q -r "$archive" "$archive_stem"

rm -rf "$archive_stem"

echo "Package created: `pwd`/$archive"
