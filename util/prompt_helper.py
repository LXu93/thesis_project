import json
import random
import os
from util import Const

def prompt_generator(pathology, template="An endoscopy image in which "):
    with open(os.path.join(os.path.abspath(os.getcwd()),'util','prompts.json')) as file:
        prompts_bank = json.load(file)
    if pathology in prompts_bank.keys():
        return template + random.choice(prompts_bank[pathology])
    return template + pathology

def standard_label(label):
    for pathology in Const.Text_Annotation.keys():
        if label in Const.Text_Annotation[pathology]:
            return pathology
    return label

