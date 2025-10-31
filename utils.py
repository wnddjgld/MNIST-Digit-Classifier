import numpy as np
from PIL import Image, ImageOps

def preprocess_image(file_obj):
    img = Image.open(file_obj).convert("L")               # 그레이스케일
    img = ImageOps.invert(img)
    img = ImageOps.pad(img, (28, 28), method=Image.BILINEAR, color=255, centering=(0.5, 0.5))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

def postprocess(pred):
    """
    예측 배열(pred)을 받아 상위 클래스와 확률,
    그리고 모든 클래스의 확률 리스트를 반환합니다.
    """
    prob = float(pred.max())
    cls = int(pred.argmax())
    
    # 10개 클래스 전체의 확률을 float 리스트로 변환
    all_probs = [float(p) for p in pred]
    
    return {"digit": cls, "prob": prob, "all_probs": all_probs}