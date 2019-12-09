#!/bin/bash


makeflow -T wq --jx main_workflow.jx --jx-args raw_50.jx $@

