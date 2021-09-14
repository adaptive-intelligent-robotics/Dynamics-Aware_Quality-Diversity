#!/usr/bin/env python3

import os
import sys
import time
import subprocess

def load_singularity_file():
    try:
        #read input file
        fin = open("singularity.def","rt")
        #print("singularity.def file found")
    
    except IOError:
        print("ERROR, singularity.def file not found!")
    finally:

        data = fin.read()
        #close the input file
        fin.close()
        return data

def get_repo_address():
    # Search projects
    command = os.popen("git config --local remote.origin.url")
    url = command.read()[:-1]
    
    if(url.startswith("git@")): #if it is using the ssh protocal, we need to convert it into an https address as the key is not available inside the container
        url=url.replace(":","/")
        url=url.replace("git@","")

    return url


def get_commit_sha_and_branch_name():
    # Search projects
    command = os.popen("git rev-parse --short HEAD")
    sha = command.read()[:-1]
    command = os.popen("git rev-parse --abbrev-ref HEAD")
    branch = command.read()[:-1]
    
    return sha,branch

def check_local_changes():
    command = os.popen("git status --porcelain --untracked-files=no")
    output = command.read()[:-1]
    if(output):
        print("WARNING: There are currently unpushed changes:")
        print(output)

def check_local_commit_is_pushed():
    command = os.popen("git rev-parse refs/remotes/origin/master")
    sha = command.read()[:-1]
    
    command = ["git", "merge-base", "--is-ancestor", sha, "HEAD"]
    output = subprocess.run(command)
    if output.returncode:
        print("WARNING: local commit not pushed, build is likely to fail!")
    
def get_repo_name():
    return os.path.basename(os.path.dirname(os.getcwd()))

def clone_commands():
    repo_address = get_repo_address()
    sha, branch = get_commit_sha_and_branch_name()
    repo_name = get_repo_name()
    
    if "CI_JOB_TOKEN" in os.environ: # we are not in a CI environment
        repo_address = f"http://gitlab-ci-token:{os.getenv('CI_JOB_TOKEN')}@{repo_address}"
    elif "PERSONAL_TOKEN" in os.environ: #if a personal token is available
        repo_address = f"https://oauth:{os.getenv('PERSONAL_TOKEN')}@{repo_address}"
    else:
        repo_address = f"https://{repo_address}"
        
        
    print(f"Building final image using branch: {branch} with sha: {sha} \n URL: {repo_address}")
    code_block = f"git clone --recurse-submodules --shallow-submodules {repo_address} {repo_name} && cd {repo_name} && git checkout {sha} && cd .."
    return code_block


def apply_changes(original_file):
    
    fout = open("./tmp.def","w")
    for line in original_file.splitlines():
        if "#NOTFORFINAL" in line:
            continue
        if "#CLONEHERE" in line:
            line = clone_commands()
        fout.write(line+"\n")
    fout.close()
    

def compile_container():
    
    image_name = f"final_{get_repo_name()}_{time.strftime('%Y-%m-%d_%H_%M_%S')}.sif"
    command = os.popen(f"singularity build --force --fakeroot {image_name} ./tmp.def && rm ./tmp.def")
    output = command.read()[:-1]
    
    
def main():
    #doing some checks and print warnings
    check_local_changes()
    check_local_commit_is_pushed()

    #getting the orignal singularity file
    data = load_singularity_file()
    #appling the changes and writing this in ./tmp.def
    apply_changes(data)
    #compiling and deleting ./tmp.def
    compile_container()
    

if __name__ == "__main__":
    main()
