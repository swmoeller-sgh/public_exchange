from torchvision.models import detection

MODEL_CATALOG_CNN = {"Fasterrcnn_resnet50": {"model": detection.fasterrcnn_resnet50_fpn,
                                             "weight": detection.FasterRCNN_ResNet50_FPN_Weights,
                                             "description": "50-layer convolutional neural network (48 convolutional layers, "
                                                            "one MaxPool layer, and one average pool layer)."},
                     "Fasterrcnn_resnet50_fpn_v2": {"model": detection.fasterrcnn_resnet50_fpn_v2,
                                                    "weight": detection.FasterRCNN_ResNet50_FPN_V2_Weights,
                                                    "description": "Constructs an improved Faster R-CNN model with a "
                                                                   "ResNet-50-FPN backbone"},
                     "FasterRCNN_MobileNet_V3_Large": {"model": detection.fasterrcnn_mobilenet_v3_large_fpn,
                                                       "weight": detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                                                       "description": "Constructs a high resolution Faster R-CNN model "
                                                                      "with a MobileNetV3-Large FPN backbone."},
                     "FasterRCNN_MobileNet_V3_Large_320": {"model": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
                                                           "weight": detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                                                           "description": "Low resolution Faster R-CNN model with a "
                                                                          "MobileNetV3-Large backbone tunned for mobile "
                                                                          "use cases."},
                     "fcos_resnet50_fpn": {"model": detection.fcos_resnet50_fpn,
                                           "weight": detection.FCOS_ResNet50_FPN_Weights,
                                           "description": "Constructs a FCOS model with a ResNet-50-FPN backbone."},
                     "RetinaNet_ResNet50": {"model": detection.retinanet_resnet50_fpn,
                                            "weight": detection.RetinaNet_ResNet50_FPN_Weights,
                                            "description": "Constructs a RetinaNet model with a ResNet-50-FPN backbone."},
                     "RetinaNet_ResNet50_V2": {"model": detection.retinanet_resnet50_fpn_v2,
                                               "weight": detection.RetinaNet_ResNet50_FPN_V2_Weights,
                                               "description": "Constructs a improved RetinaNet model with a ResNet-50-FPN "
                                                              "backbone."},
                     "SSD300_VGG16": {"model": detection.ssd300_vgg16,
                                      "weight": detection.SSD300_VGG16_Weights,
                                      "description": "The SSD300 model is based on the SSD: Single Shot MultiBox "
                                                     "Detector paper."},
                     "SSDLite320_MobileNet_V3_Large": {"model": detection.ssdlite320_mobilenet_v3_large,
                                                       "weight": detection.SSDLite320_MobileNet_V3_Large_Weights,
                                                       "description": "SSDlite model architecture with input size 320x320 "
                                                                      "and a MobileNetV3 Large backbone, as described at "
                                                                      "Searching for MobileNetV3 and MobileNetV2: Inverted "
                                                                      "Residuals and Linear Bottlenecks."},
                     "Test all models": {}
                     }
