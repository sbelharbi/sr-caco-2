#!/usr/bin/env bash

# super-resolution
echo 'storing datasets at: ' $1
# DIV2K
#wget -P $1 https://cv.snu.ac.kr/research/EDSR/DIV2K.tar

# FLicker2K
wget -P $1 https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar

# Test datasets
echo 'https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u'