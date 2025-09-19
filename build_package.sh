#!/bin/bash

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
# Date:     2025-06-23

cd $(dirname "$0")

package_path="cslics_vision_processor"
include_directories=("$package_path" "services" "config")
include_files=(LICENCE.txt requirements.txt client_vision_processor.py)

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
