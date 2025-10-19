from ultralytics import YOLO

class model_yolo:
    def __init__(self):
        # Gerekli ağırlık dosyalarını ekliyoruz  
        self._road_seg_model = YOLO("weights/road_segmentation.pt")
        self._car_model = YOLO("weights/yolo11n.pt")
        self._line_detect = YOLO("weights/line_detection.pt")
        # Modellerin gpu'da çalışması için cuda çalıştırıyoruz 
        self._car_model.to("cuda").half()
        self._road_seg_model.to("cuda").half()
        self._line_detect.to("cuda").half()

    def road_segmentation(self):
        return self._road_seg_model
    def car_detect(self):
        return self._car_model
    def line_detect(self):
        return self._line_detect