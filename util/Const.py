from torchvision import transforms

All_Datasets = ["CAPTIV8", 
               "SEE-AI", 
               "KID2", 
               "Kvasir-Capsule", 
               "WCEBleedGen",
               "SUN",
               "Kvasir v2",
               "Mark-data"]

WCE_Datasets = ["CAPTIV8", 
                "SEE-AI", 
                "KID2", 
                "Kvasir-Capsule", 
                "WCEBleedGen",
                "Mark-data"]

COLN_Datasets = ["Kvasir v2",
                 "SUN"]

Text_Annotation = {"polyp": ["polyp", 
                             "polyps",
                             "Polyp",
                             "polypoids", 
                             "polyp-like",
                             "pseudopolyp"],
                    "non-polyp": ["non-polyp",
                                  "normal"],
                    "inflammation": ["inflammation",
                                     "Inflammation",
                                     "inflammatory change",
                                     "inflammatory",
                                     "erosion",
                                     "Erosion"],
                    "erosion": ["erosion",
                                "Erosiom"],
                    "bleeding": ["bleeding",
                                 "Bleeding",
                                 "blood_fresh",
                                 "blood"],
                    "bubbles":["bubbles"],
                    "dirt": ["dirt"],
                    "clean": ["clean"],
                    "dirt_and_bubbles":["dirt_and_bubbles"],
                    "technical": ["bubbles",
                                  "dirt",
                                  "clean",
                                  "dirt_and_bubbles"]
                    
                   }


PREPROCESSOR = {
    "general": transforms.Compose([
                transforms.ToTensor(),               
            ]),
    "CAPTIV8": transforms.Compose([
                transforms.CenterCrop((512,512)),
                transforms.ToTensor(),               
            ]),
    "Mark-data": transforms.Compose([
                transforms.CenterCrop((512,512)),    
                transforms.ToTensor(),               
            ]),
    "SEE-AI": transforms.Compose([
                transforms.CenterCrop((512,512)),    
                transforms.ToTensor(),               
            ]),
    "KID2": transforms.Compose([
                transforms.CenterCrop((320,320)),  
                transforms.ToTensor(),               
            ]),
}