import math  # Matematiksel işlemler için math kütüphanesi

class Tracker:
    def __init__(self):
        # Nesnelerin merkez pozisyonlarını saklamak için bir sözlük oluşturulur
        self.center_points = {}
        # Nesne kimliklerinin sayısını tutmak için bir sayaç oluşturulur
        # Her yeni bir nesne tespit edildiğinde, sayaç bir artırılır
        self.id_count = 0

    def update(self, objects_rect):
        # Nesnelerin kutuları ve kimlikleri
        objects_bbs_ids = []

        # Yeni nesnenin merkez noktasını bulunur
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2  # X koordinatı merkezi
            cy = (y + y + h) // 2  # Y koordinatı merkezi

            # Bu nesnenin zaten tespit edilip edilmediğini kontrol eder
            ayni_nesne_tespit_edildi = False
            for id, pt in self.center_points.items():
                # Merkez noktaları arasındaki mesafeyi hesaplar
                dist = math.hypot(cx - pt[0], cy - pt[1])

                # Eğer mesafe belli bir eşiği aşıyorsa, aynı nesne kabul edilir
                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    ayni_nesne_tespit_edildi = True
                    break

            # Eğer yeni bir nesne tespit edilmediyse, ID bu nesneye atanır
            if ayni_nesne_tespit_edildi is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Artık kullanılmayan ID'leri temizlemek için merkez noktaları sözlüğü güncellenir
        yeni_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            yeni_center_points[object_id] = center

        # Kullanılmayan ID'leri temizlenmiş olan merkez noktaları sözlüğü ile güncellenir
        self.center_points = yeni_center_points.copy()
        return objects_bbs_ids
