#!/bin/bash 

ls *.tar | xargs -I {} tar -xvf {}
