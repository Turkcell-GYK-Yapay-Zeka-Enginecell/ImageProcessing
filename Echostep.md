# EchoCell Dataset
- [Coco Dataset](https://cocodataset.org/#home)
<i>
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
</i>

- [Open Images Datase](https://storage.googleapis.com/openimages/web/index.html)
- [Indoor Scene Recognition](https://web.mit.edu/torralba/www/indoor.html)
- [Indoor Scene](https://rgbd.cs.princeton.edu/)
- [Indoor Scene](https://objectnet.dev/)
- [Kitchen Scene](https://universe.roboflow.com/sdc-gz79b/ai2thor)
## Road Dataset
- [Road](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k/data)
<i> person(129262), rider(6461), car(1021857), truck(42963), bus(16505), train(179), motor (motorcycle)(4296), bike (bicycle)(10229), traffic light(265906), traffic sign(343777)
 </i>
 
- [Road Sign](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection)
  <i>Trafic Light, Stop, Speedlimit, Crosswalk  (877) </i>
  
- [Road Obstacles](https://data.mendeley.com/datasets/jr3yv3wfhx/1) <i> Auto,Crosswalk,Electric Bike,Random,Road Obstacle,Road sign,Road Traffic,Road,Tractor,Traffic Sign,Truck Tempo
</i>

- [Road Obstacles 2](https://universe.roboflow.com/safewalkbd/safewalkbd-l8jbn)
<i> Animal(1682), Crosswalk, Obstacle(3249), Over-bridge, Person(4722), Pole(4533), Pothole(1568), Railway, Road-barrier, Sidewalk(1469), Stairs(1375), Traffic-light(1216), Traffic-sign, Train, Tree(1698), Vehicle(7082)
</i>

- [Road Obstacles 3](https://universe.roboflow.com/scottsdale/sidewalk-otzhb)
<i>Animal, Crosswalk, Obstacle, Over-bridge, Person, Pole, Pothole, Railway, Road-barrier, Sidewalk, Stairs, Traffic-light(487), Traffic-sign(697), Train, Tree, Vehicle </i>

- [Road Obstacles 4](https://universe.roboflow.com/turkeyroadsigns/turkey-road-sign?utm_source=chatgpt.com)
<i> 20, 30, dur, durak, girisyok, ilerisag, ilerisol, kirmizi, park, parkyasak, sag, sagadonulmez, sari, sol, soladonulmez, yesil
 </i>


## CLASSES
### Emine
- İnsan / Yaya (Person, Pedestrian)
- Bisiklet (Bicycle)
- Araçlar (Car, Truck, Bus, Motorcycle)
- Trafik ışığı (Traffic Light)
- Trafik işaretleri (Traffic Sign, Stop Sign, Speed Limit)
- Kaldırım (Sidewalk)

- ## Coco

- İnsan / Yaya (Person, Pedestrian) 66.808
- Bisiklet (Bicycle) 3.401
- Araçlar (Car, Truck, Bus, Motorcycle) 12.786 araba, 3.661 motosiklet, 3745 tren, 6377 kamyon,
- Trafik ışığı (Traffic Light) 4330
- Trafik işaretleri (Traffic Sign, Stop Sign, Speed Limit) , dur işareti 1.803, 
- Kaldırım (Sidewalk)

- ## Sidewalk 

- İnsan / Yaya (Person, Pedestrian) /747
- Bisiklet (Bicycle) / 156
- Araçlar (Car, Truck, Bus, Motorcycle)  327 otobüs, 1.674 araba, 38 motosiklet, 809 kamyon
- Trafik ışığı (Traffic Light) / 243 trafik ışıkları
- Trafik işaretleri (Traffic Sign, Stop Sign, Speed Limit) /  487 trafik işareti, 697 trafik 
- Kaldırım (Sidewalk)/ 
- Engel 446 +100
- Ağaç 2.370 
- Direk 2.236


- ## Turkey road sign 

- İnsan / Yaya (Person, Pedestrian) 66.808
- Bisiklet (Bicycle) 
- Araçlar (Car, Truck, Bus, Motorcycle)  
- Trafik ışığı (Traffic Light) 
- Trafik işaretleri  14.717 trafik işaretleri , kırmızı ışık 687 , yeşil ışık 533
- Kaldırım (Sidewalk)


### Zuhal
- Yol (Road, Street)
  - [Bu datasette düz yol(367) ve çukur(357) görüntüleri var.](https://www.kaggle.com/datasets/virenbr11/pothole-and-plain-rode-images)
  - [Yol görünrüleri 2000+](https://www.kaggle.com/datasets/dataclusterlabs/lane-detection-road-line-detection-image-dataset)
- Engel / Engel çeşitleri (Obstacle, Barrier, Construction Cone)
  - [Bu datasette road(586), pothole(810), yavaşlama engeli(306) var](https://www.kaggle.com/datasets/shrunmayshinde/road-obstacles-detection)
  - [Bu datasette 646 construction code vardır](https://universe.roboflow.com/robotica-xftin/traffic-cones-4laxg/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
- Yol çizgileri (Crosswalk, Lane Marker)
  - [Bu datasette yaya geçidi görslleri var.](https://github.com/xN1ckuz/Crosswalks-Detection-using-YOLO/tree/main)
  - [Datasette 270 yayageçidi var](https://universe.roboflow.com/tfg-7qtpm/accesibility-street/browse?queryText=class%3Acrosswalk&pageSize=50&startingIndex=0&browseQuery=true)
- Yol kenarı (Road Edge)
  - [Bu datasette street ve ](https://universe.roboflow.com/data-dynamos/streets-and-crosswalks)
- Çöp kutusu (Trash Can)
  - [Bu datasette sokakta bulunan çöp görüntüleri var.12752](https://universe.roboflow.com/garbage-wbsv6/plitter)
- Ağaçlar / Bitkiler (Tree, Bush)
  - [Bu datasette çalı görüntüleri(361 var.](https://universe.roboflow.com/taiganguyen/obstacle-detecting/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
 

### Gülizar
- Elektrik direği (Pole)
    - [Road Obstacles 2](https://universe.roboflow.com/safewalkbd/safewalkbd-l8jbn) - 4533
    - [Road Obstacles 3](https://universe.roboflow.com/scottsdale/sidewalk-otzhb) - 2236
- Merdiven (Stairs)
    - [Road Obstacles 2](https://universe.roboflow.com/safewalkbd/safewalkbd-l8jbn) - 1375
    - [stairs](https://universe.roboflow.com/perception-01-3e0d2/stairs-zz3xs) - 1511
- Toprak veya çimen alanlar (Grass, Dirt)
    - [grass](https://universe.roboflow.com/grass-iaytu/grass-gp8zy) - 1147
- Park alanı (Parking Spot) ?
- Yönlendirme tabelaları (Directional Sign)
   - [Road Obstacles 4](https://universe.roboflow.com/turkeyroadsigns/turkey-road-sign?utm_source=chatgpt.com)
   - soladonulmez - 3718
   - girisyok - 3666
   - durak - 2004
   - sagadonulmez - 1958
   - park - 1711
   - kirmizi - 1145
   - parkyasak - 964
   - dur - 948
   - ilerisol - 886
   - ilerisag - 838 
- Engelli rampası (Wheelchair Ramp)
   - [wheelchair ramp](https://universe.roboflow.com/ramp-5cb74/ramp-ync8p)  - 4032
