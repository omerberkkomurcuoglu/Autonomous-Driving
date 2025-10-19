import carla
import random
import time
import numpy as np
import cv2
from threading import Thread, Lock ,Event
from collections import deque
from queue import Queue
from Model_Yolo import model_yolo
from Setup import CarlaSetup

IM_HEIGHT = 480
IM_WIDTH = 640

# Simülasyon'u cmd'de çalıştırma komutu 
# CarlaUnreal.exe -ResX=10 -ResY=10 quality-level=low -windowed

frame_queue = Queue(maxsize=10)    # Kameradan gelen frame'ler buraya
results_queue = deque(maxlen=3)  # Predict sonucu ve frame burada
stop_event = Event()    # Tüm thread'lerin kontrolü için

# !! Simülasyon'dan gelen frameleri doğru formata dönüştürüp queue'e atıyor çağırıldığı yer (189 satır)
def process_img(frame):
    # Gerekli dönüşümler tutorialdan baktım 
    array = np.frombuffer(frame.raw_data, dtype=np.uint8) 
    array = array.reshape((IM_HEIGHT, IM_WIDTH, 4))
    img = array[:, :, :3]
    try:
        # Bir queue oluşturup frameleri ona atıyoruz 
        frame_queue.put_nowait(img.copy())
    except:
        # Eğer queue doluysa frame atla
        pass

model_road_segment = None
model_car_detect = None
model_line_detect = None
class Inference:
    def __init__(self):
        # İki thread'in aynı frame'i alabilmesi için results_queue(Deque) sunda hangi thread'in en son aldığı eleman id'sini tutuyor 
        self.last_ids = {"display": -1, "move": -1} 
        self.frame_id = 0
        self.lock = Lock()
    """
    Frame'leri modellerle tahmin ettiriyoruz ve results_queue'suna atıyoruz 
    Kısaca tahmin işlemleri yapılıyor 
    """
    def inference_loop(self):
        i= 0
        while not stop_event.is_set():  # thread döngüsü 
            try:
                img =frame_queue.get_nowait() # frame_queue dan baştaki frame'i alır 
            except:
                continue
            
            print("inference ",i)
            i+=1
            results_car,results_road,results_line = None,None,None
            
            prev =time.time()
            # Tahmin işlemleri gerçekleştiriliyor
            results_car = model_car_detect(img,verbose=False)[0]  
            results_line = model_line_detect(img,verbose=False)[0]
            results_road = model_road_segment(img,verbose=False)[0]
            curr = time.time()
            print("Sure Tahmin :",curr-prev) # Tahmin'in süresini görüyoruz 
            with self.lock:
                # Lock atıp results_queue a tahmin değerlerini ve frame_id sini kaydediyoruz 
                results_queue.append((self.frame_id, img, results_car, results_road,results_line))
                self.frame_id += 1

    def get_latest_frame(self, role):
        with self.lock: 
            # Display ve move threadleri son thread'i alır 
            if not results_queue:  # Deque da eleman yoksa her şeyi None döndürür
                return None, None, None, None,None
            fid, img, results_car, results_road,results_line = results_queue[-1] # Deque daki son framei alıyoruz 
            if fid == self.last_ids[role]: # Eğer bir thread (move veya display) hali hazırda bu frame'i almışsa  none döndürüyoruz 
                return None, None, None,None,None
            self.last_ids[role] = fid # frame_id i hangi thread aldıysa onun last_ids sözlüğündeki value'suna eşitliyoruz 
            return fid, img, results_car, results_road,results_line  # Değerleri dönduruyoruz 

inference = Inference()
class Display:
    def __init__(self, yolo_model):
        self.yolo_model = yolo_model
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
    """
    Bu fonksiyon kullanıcıya sonuçları görsel olarak gösterilmesine yarıyor 
    """
    def display_loop(self, model_car, model_road):
        prev_time = time.time()
        i = 0

        # VideoWriter ayarları
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))


        while True:
            # Display threadi en son eklenen frame ve tahmin değerlerini alıyor 
            fid, img, result_car, result_road,result_line = inference.get_latest_frame("display") 
            if img is None: # Sonuç yoksa küçük bir uyku cpu rahatlatmak için 
                time.sleep(0.001)
                continue
            print("Display ",i) # Tamamen kontrol amaçlı 
            i+=1
            # Fps hesabı 
            self.frame_count += 1  # Frame per second hesabı yapılıyor 1 saniyede kaç frame'in işlendiğini görüyoruz 
            if self.frame_count % 30 == 0:
                curr_time = time.time()
                self.fps = 30.0 / (curr_time - self.last_fps_time)
                self.last_fps_time = curr_time
            # 3 tane tahminimiz var şimdilik hangisi varsa onlara uygun if ve arıdndan method çalışıyor
            if result_car: 
                self.draw_boxes(img, result_car, model_car) # Bounding_box'a alıyor tahmin edilen nesneyi
            if result_road:
                # Segmentasyon yapıyor, yani yol olarak gördüğü yerlere maske koyuyoruz 
                img = self.add_weights(img, result_road,1) 
            if result_line:
                
                img =self.add_weights(img,result_line,2)
            if img.size == 0:
                continue
            
            cv2.putText(img, f"FPS: {self.fps:.2f}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # Fps'i ekrana yazdırır
            cv2.imshow("Camera View", img)  # Görünmtüyü ekrana getiriyoruz 
            out.write(cv2.resize(img, (640, 480)))
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                print("Kayıt durduruldu ve video kaydedildi.")
                stop_event.set() # Threadleri durdurur         
                print("Kayıt durduruldu ve video kaydedildi.")
                return False
    def draw_boxes(self, img, results, model, threshold=0.4):
        # Gelen box'ları ham görüntüye ekliyoruz 
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if conf > threshold:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def add_weights(self, img, results,color):
        # Segmentation için gelen verilerden mask çıkarıp ham görüntüye ekliyoruz 
        mask = results.masks.data[0].cpu().numpy()
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask_uint8, (img.shape[1], img.shape[0]))
        color_mask = np.zeros_like(img)
        color_mask[:, :, color] = mask_resized
        img = cv2.addWeighted(img, 1, color_mask, 0.5*color, 0)
        return img
    
class CarControl:
    """
      Burada çizgi ve yol maskelerine ve lidar sensorlerine bakarak araç yönlendirilmeye çalişilacak
      (PID kontroller kullanilarak)
      Şimdilik sadece dümdüz hareket ettiriliyor 
    """
    
    def __init__(self, yolo_model):
        self.yolo_model = yolo_model
    
    def move_car(self, vehicle, model_car):
    
        print("Araç hareket ediyor...")
        i = 0
        while not stop_event.is_set():
            print("Move Car ",i)
            time.sleep(0.1)
            i+=1
            vehicle.apply_control(carla.VehicleControl(throttle=0.1, steer=0.0)) # Aracı hareket ettiriyor 
            fid, img, result_car, result_road,result_line= inference.get_latest_frame("move") # En son thread'i alıyoruz 
            if result_car is None:
                continue
            else:
                # Araç algılama sonucu geldiyse araç kontrolü burada yapılabilir 
                print(f"Move thread - Frame ID: {fid}, Detected {len(result_car.boxes)} vehicles")
                

def main():
    # Gerekli class ve methodları main'de çağırıyoruz 
    yolo_model = model_yolo()
    car_control = CarControl(yolo_model)
    display = Display(yolo_model)
    global model_road_segment, model_car_detect,model_line_detect
    model_road_segment = yolo_model.road_segmentation()
    model_car_detect = yolo_model.car_detect()
    model_line_detect = yolo_model.line_detect()
    
    carlaSet = CarlaSetup()
    try:
        spawn_points = carlaSet.spawnPoints()  # CarlaSetup'tan spawn noktalarını çağırıyoruz 
        spawn_point = random.choice(spawn_points) #  Bu noktalardan rastgele bir nokta seçiyoruz
        vehicle = carlaSet.addVehicle("vehicle.dodgecop.charger", spawn_point) # CarlaSetup'a gerekli parametreleri yollayarak aracı spawnlıyoruz 

        camera = carlaSet.addCamera("sensor.camera.rgb", IM_WIDTH, IM_HEIGHT, 110, 1.5, 2.4, vehicle)  # Camera spawnlıyoruz 
        camera.listen(lambda data: process_img(data)) # Kameradan görüntü alıyoruz 
        
        inference_thread = Thread(target=inference.inference_loop)  # İnference threadini tanımlıyoruz  
        move_thread = Thread(target=car_control.move_car, args=(vehicle, model_car_detect)) # Move threadini tanımlıyoruz  
        # Threadleri başlatıyoruz 
        inference_thread.start()
        move_thread.start()
        # Display ana thread’de çalışsın
        display.display_loop(model_car_detect, model_road_segment)

        inference_thread.join()
        move_thread.join()

    finally:
        print("Temizlik yapılıyor...")
        for actor in carlaSet.actor_list:
            actor.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
