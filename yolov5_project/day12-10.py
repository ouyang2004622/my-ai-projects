import torch
model=torch.hub.load("D:/yolov5-7.0","custom",path="./yolov5s.pt",source="local")
img="D:/yolov5-7.0/data/images/IMG_5183.jpg"
result=model(img)
result.show()