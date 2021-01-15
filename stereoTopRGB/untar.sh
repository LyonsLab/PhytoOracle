#!/bin/bash 

ls *_plotclip.tar | xargs -I {} tar -xvf {}
