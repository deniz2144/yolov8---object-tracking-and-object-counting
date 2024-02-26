import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker

model = YOLO('yolov8s.pt')
# İzleme ve sayım alanlarının koordinatları belirlenir
area1 = [(312, 388), (289, 390), (474, 469), (497, 462)] # İçeri giriş alanı
area2 = [(279, 392), (250, 397), (423, 477), (454, 469)] # Dışarı çıkış alanı 

# Fare olaylarını dinleyen bir fonksiyon tanımlanır
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  # Eğer fare hareket ederse
        colorsBGR = [x, y] # Fare konumunu al
        print(colorsBGR)  
# RGB penceresi oluşturulur ve fare olayları bu pencerede izlenir        
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('C:\\Users\\deniz\\Downloads\\peoplecounteryolov8-main (1)\\peoplecounteryolov8-main\\peoplecount1.mp4')

my_file = open("C:\\Users\\deniz\\Downloads\\peoplecounteryolov8-main (1)\\peoplecounteryolov8-main\\coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")# Sınıf listesi bir dize listesine dönüştürülür

# Sınıf listesi bir dize listesine dönüştürülür
count = 0

tracker = Tracker()# Nesne izleyici nesnesi
people_entering = {}# İçeri giren insanlar için sözlük
entering = set()# İçeri giren insanların kimlikleri
people_exiting = {}# Dışarı çıkan insanlar için sözlük
exiting = set()  # Dışarı çıkan insanların kimlikleri

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
        # Her iki karede bir işlem yapılır (FPS düşürmek için)

    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    # YOLO modeli ile nesne tanıma yapılır
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []
     #YOLO tarafından tespit edilen nesnelerin koordinatları alınır         
    for index, row in px.iterrows():
        # Sınırlayıcı kutunun sol üst köşesinin x koordinatı
        x1 = int(row[0])
        # Sınırlayıcı kutunun sol üst köşesinin y koordinatı
        y1 = int(row[1])
        # Sınırlayıcı kutunun sağ alt köşesinin x koordinatı
        x2 = int(row[2])
        # Sınırlayıcı kutunun sağ alt köşesinin y koordinatı
        y2 = int(row[3])
        # Nesnenin sınıfının indeksi
        d = int(row[5])
        # Nesnenin sınıfının adı
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])

    # İzleme nesnesi güncellenir. Nesnelerin sınırlayıcı kutularını ifade eder. 
       
    bbox_id = tracker.update(list)
        # Her nesne için giriş ve çıkış izlenir
    for bbox in bbox_id:
          # Sınırlayıcı kutunun sol üst köşesinin x koordinatı, y koordinatı, sağ alt köşesinin x koordinatı, y koordinatı ve nesnenin ID'si
        x3, y3, x4, y4, id = bbox
        # Eğer nesne alan2(area2) içinde tespit edilmişse içeri girmiş olarak işaretlenir
        #Bu satır, bir noktanın bir çokgenin içinde olup olmadığını kontrol eder. Bu durumda, nokta x4, y4 (nesnenin sınırlayıcı kutusunun bir köşesi) ve çokgen area2dir. Fonksiyon, nokta çokgenin içindeyse 0’dan büyük bir değer döndürür.
        results = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
        if results >= 0:#Bu satır, eğer nesne area2 içinde tespit edilmişse bir işlem yapar.
            people_entering[id] = (x4, y4)# Bu satır, nesnenin (kişinin) son konumunupeople_entering sözlüğüne ekler. Bu bilgi, daha sonra nesnenin (kişinin) hareketlerini izlemek için kullanılabilir.
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0,255), 2)#: Bu satır, nesnenin (kişinin) sınırlayıcı kutusunu çizer. (0, 0, 255) BGR renk kodu kırmızıdır. Bu, nesnenin (kişinin) video karesi üzerindeki konumunu görsel olarak belirtir. Bu, örneğin bir video üzerindeki kişileri izlemek ve belirli alanlara giriş ve çıkışları saymak için kullanılabilir. Bu kod parçası, video analitiği ve nesne izleme uygulamalarında yaygın olarak kullanılır.
 

        # Bu satır, eğer nesnenin (kişinin) ID’si people_entering sözlüğünde varsa bir işlem yapar. Bu, nesnenin (kişinin) daha önce area2 içinde tespit edildiğini ve içeri girmiş olarak işaretlendiğini belirtir.

            
        if id in people_entering:
           #bir noktanın bir çokgenin içinde olup olmadığını kontrol eder. Bu durumda, nokta x4, y4 (nesnenin sınırlayıcı kutusunun bir köşesi) ve çokgen area1dir. Fonksiyon, nokta çokgenin içindeyse 0’dan büyük bir değer döndürür.
           results1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
           #if results1>= 0:: Bu satır, eğer nesne area1 içinde tespit edilmişse bir işlem yapar.

           
           if results1>= 0:
              # Bu satır, nesnenin (kişinin) sınırlayıcı kutusunu çizer. (0, 255, 0) BGR renk kodu yeşildir. Bu, nesnenin (kişinin) video karesi üzerindeki konumunu görsel olarak belirtir.
              cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
              #: Bu satır, nesnenin (kişinin) konumunda bir daire çizer. (255, 0, 255) BGR renk kodu mor renktir. Bu, nesnenin (kişinin) video karesi üzerindeki konumunu görsel olarak belirtir.
              cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
              cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
              # Bu satır, nesnenin (kişinin) ID’sini entering setine ekler. Bu, nesnenin (kişinin) area1 içine girdiğini belirtir. Bu bilgi, daha sonra nesnenin (kişinin) hareketlerini izlemek için kullanılabilir. Bu kod parçası, video analitiği ve nesne izleme uygulamalarında yaygın olarak kullanılır.
              entering.add(id)
            ##exiting
        #ir noktanın bir çokgenin içinde olup olmadığını kontrol eder. Bu durumda, nokta x4, y4 (nesnenin sınırlayıcı kutusunun bir köşesi) ve çokgen area1dir. Fonksiyon, nokta çokgenin içindeyse 0’dan büyük bir değer döndürür.
        results2 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)

        if results2>= 0:
               
               people_exiting[id] = (x4, y4)
               #Bu satır, nesnenin (kişinin) sınırlayıcı kutusunu çizer. (0, 255, 0) BGR renk kodu yeşildir. Bu, nesnenin (kişinin) video karesi üzerindeki konumunu görsel olarak belirtir.
               cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        # satır, eğer nesnenin (kişinin) ID’si people_exiting sözlüğünde varsa bir işlem yapar. Bu, nesnenin (kişinin) daha önce area1 içinde tespit edildiğini ve dışarı çıkmış olarak işaretlendiğini belirtir.
        if id in people_exiting:
            #bir noktanın bir çokgenin içinde olup olmadığını kontrol eder. Bu durumda, nokta x4, y4 (nesnenin sınırlayıcı kutusunun bir köşesi) ve çokgen area2dir. Fonksiyon, nokta çokgenin içindeyse 0’dan büyük bir değer döndürür.

            results3 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
            if results3 >= 0:
                    #nesnenin (kişinin) sınırlayıcı kutusunu çizer. (255, 0, 255) BGR renk kodu mor renktir. Bu, nesnenin (kişinin) video karesi üzerindeki konumunu görsel olarak belirtir.
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                    #Bu satır, nesnenin (kişinin) konumunda bir daire çizer. (255, 0, 255) BGR renk kodu mor renktir. Bu, nesnenin (kişinin) video karesi üzerindeki konumunu görsel olarak belirtir.
                    cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                    # nesnenin (kişinin) ID’sini video karesi üzerine yazdırır. Bu, nesnenin (kişinin) benzersiz ID’sini görsel olarak belirtir.
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
                    #nesnenin (kişinin) ID’sini exiting setine ekler.
                    exiting.add(id)
    #area1 çokgenini çizer. frame üzerinde, area1 koordinatlarına sahip bir çokgen çizer. (255, 0, 0) BGR renk kodu mavi renktir. Bu, area1in video karesi üzerindeki konumunu görsel olarak belirtir.
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    #1’ metnini video karesi üzerine yazdırır. Bu, area1in etiketini görsel olarak belirtir.
    cv2.putText(frame, str('1'), (504, 471), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
   #area2 çokgenini çizer. frame üzerinde, area2 koordinatlarına sahip bir çokgen çizer. (255, 0, 0) BGR renk kodu mavi renktir. Bu, area2in video karesi üzerindeki konumunu görsel olarak belirtir.
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    #Bu satır, ‘2’ metnini video karesi üzerine yazdırır. Bu, area2in etiketini görsel olarak belirtir.
    cv2.putText(frame, str('2'), (466, 485), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
    #entering setindeki elemanların sayısını hesaplar. Bu, area1e giren kişi sayısını belirtir.
    İ=(len(entering))
    #Bu satır, exiting setindeki elemanların sayısını hesaplar. Bu, area2den çıkan kişi sayısını belirtir.
    O=(len(exiting))
    #Bu satır, area1e giren kişi sayısını video karesi üzerine yazdırır. Bu, area1e giren kişi sayısını görsel olarak belirtir.
    cv2.putText(frame, str(İ), (60, 80), cv2.FONT_HERSHEY_COMPLEX,(0.7), (0, 0,255),2)
    #Bu satır, area2den çıkan kişi sayısını video karesi üzerine yazdırır. Bu, area2den çıkan kişi sayısını görsel olarak belirtir.
    cv2.putText(frame, str(O), (60, 140), cv2.FONT_HERSHEY_COMPLEX, (0.7), (255, 0,255),2)

    
    cv2.imshow("RGB", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()