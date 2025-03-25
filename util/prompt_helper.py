import json
import random
import os
from util import Const

def prompt_generator(pathology, section="unknown", template="An endoscopy image"):
    with open(os.path.join(os.path.abspath(os.getcwd()),'util','prompts.json')) as file:
        prompts_bank = json.load(file)
    if section == "random":
        section = random.choice(Const.Text_Annotation["section"])
    elif section != "unknown":
        template = f"{template} in {section}," 
    if pathology in prompts_bank.keys():
        return f"{template} {random.choice(prompts_bank[pathology])}"
    return template + pathology

def standard_label(label):
    for pathology in Const.Text_Annotation.keys():
        if label in Const.Text_Annotation[pathology]:
            return pathology
    return label

