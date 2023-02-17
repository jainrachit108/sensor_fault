from setuptools import find_packages,setup

from typing import List

REQUIREMENT_FILE = "requirements.txt"
HYPHEN_E_DOT = "-e ."

def get_requirements()->List[str]:
    with open(REQUIREMENT_FILE) as requirement_file:
        requirement_list = requirement_file.readlines() 
    requirement_list = [requirement_name.replace('\n', '') for requirement_name in requirement_list]

    if HYPHEN_E_DOT in requirement_list:
        requirement_list.remove(HYPHEN_E_DOT)
    
    return requirement_list



setup(
    name="sensor",
    version="0.0.1",
    author="Rachit",
    author_email="jainrachit124@gmail.com",
    packages = find_packages(),
    install_requires=get_requirements(),
)