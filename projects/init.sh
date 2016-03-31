#!/bin/bash

name=$1

if [[ -n "$name" ]]; then
        cp -r hello_world "$name"
        cd "$name"
        mv src/main_hello_world.cpp src/main_"$name".cpp
        mv hello_world.vcxproj "$name".vcxproj
        sed -e 's/hello_world/'"$name"'/g' Makefile > Makefile.tmp
        mv Makefile.tmp Makefile
        git add *.vcxproj Makefile inc/.readme src/.readme src/main* scripts/*.m results/.readme
        cd ..
else
        echo "Usage: ./init.sh <project_name>"
fi
