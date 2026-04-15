from roboflow import Roboflow
rf = Roboflow(api_key="SWqWXkWHv3qjkrFsZRIp")
project = rf.workspace("laparoscopic-yolo").project("laparoscopy")
version = project.version(14)
dataset = version.download("yolov8-obb", location = "data/surgical-instruments")
                