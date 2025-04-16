# -*- coding: utf-8 -*-
"""config.py
by fcascan 2025
"""
import os

MAX_CAMERAS_TO_SCAN = 6
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "assets/models/yolo_quant_int8.rknn")
IMG_SIZE = (640, 640)
FPS_TEXT_SIZE = 0.5
LABEL_TEXT_SIZE = 0.4

# COCO dataset; change for yours (if custom dataset used)
##CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
##            "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
##            "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
##            "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
##            "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
##            "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
##            "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")
CLASSES = ("person", "gun", "pistol", "revolver", "firearm", "knife", "machete", "katana", "dagger", "sword", "axe", "bat", "club", 
           "bludgeon", "stick", "nunchaku", "shuriken", "throwing star", "crossbow", "spear", "harpoon", "boomerang", 
           "slingshot", "catapult", "grenade", "explosive device", "machine gun", "shotgun", "assault rifle", "sniper rifle", "submachine gun",
            "light machine gun", "heavy machine gun", "rocket launcher", "missile launcher", "flamethrower", "taser", "stun gun", "pepper spray")