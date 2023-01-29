"""
Objective
==============
Generate the caption for an image using two NN (CNN and NLP).
"""


# Library import
# ===============
import torchvision.models


# Variable definition
# ===================


# CLASS definition
# ================


# FUNCTION definition
# ===================


# MODEL definition
# ================
frcnn_model = torchvision.models.get_model("fasterrcnn_resnet50_fpn", weights="DEFAULT")


# ===================
# MAIN
# ===================
print("[INFO] Parameter of selected model for image recognition", frcnn_model)