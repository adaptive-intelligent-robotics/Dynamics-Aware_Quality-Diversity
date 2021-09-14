#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import argparse

DEFNAME="singularity.def"
EXP_PATH="/git/sferes2/exp/"

def get_image_name():
    return f"{os.path.basename(os.path.dirname(os.getcwd()))}_v2_ffmpeg.sif"


def build_sandbox():
    #check if the sandbox has already been created
    image_name = get_image_name()
    if os.path.exists(image_name):
        return
    
    print(f"{image_name} does not exist, building it now from {DEFNAME}")
    assert os.path.exists(DEFNAME) #exit if defname is not found

    #run commands
    command = os.popen(f"singularity build --force --fakeroot --sandbox {image_name} {DEFNAME}")
    output = command.read()[:-1]
    


    
def run_container(nvidia): 
    image_name = get_image_name()
    
    additional_args = ""
    if nvidia:
        print("Nvidia runtime ON")
        additional_args += "--nv"

    #command = f"singularity shell -w {image_name}"
    command = f"singularity shell {additional_args} --bind {os.path.dirname(os.getcwd())}:{EXP_PATH}/{image_name[:-4]} {image_name}"
    subprocess.run(command.split())
    
    
def main():

    parser = argparse.ArgumentParser(description='Build a sandbox container and shell into it.')
    parser.add_argument('-n', '--nv', action='store_true', help='enable experimental Nvidia support')
    
    args = parser.parse_args()

    build_sandbox()
    run_container(args.nv)
    

if __name__ == "__main__":
    main()
