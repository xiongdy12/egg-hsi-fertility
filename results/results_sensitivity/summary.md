# Analisis sensitivitas tanpa telur bertanda

Telur dikeluarkan: 57

## A. Prediksi lama (dilatih dengan semua telur), dievaluasi pada semua vs bersih
PLS-DA SNV                 | semua n=1228: bal_acc=0.855 AUC=0.930 | bersih n=1171: bal_acc=0.852 AUC=0.928 | bertanda n=57: bal_acc=0.920 AUC=0.971
Kontrol spektral (5 seed)  | semua n=1228: bal_acc=0.814 AUC=0.884 | bersih n=1171: bal_acc=0.809 AUC=0.881 | bertanda n=57: bal_acc=0.870 AUC=0.915
YOLOv11s (5 seed)          | semua n=1228: bal_acc=0.845 AUC=0.927 | bersih n=1171: bal_acc=0.841 AUC=0.924 | bertanda n=57: bal_acc=0.894 AUC=0.979
Fusi PLS + YOLO            | semua n=1228: bal_acc=0.878 AUC=0.944 | bersih n=1171: bal_acc=0.874 AUC=0.942 | bertanda n=57: bal_acc=0.957 AUC=0.977
Pada telur bersih: fusi vs PLS -> dAUC=+0.0146 [+0.0073,+0.0225] P(<=0)=0.0002 | dBalAcc=+0.0219 [+0.0037,+0.0398] P(<=0)=0.0096 | McNemar p=0.0287 (73/48)
Pada telur bersih: YOLO vs kontrol spektral -> dAUC=+0.0429 [+0.0269,+0.0588] P(<=0)=0.0000 | dBalAcc=+0.0320 [+0.0086,+0.0556] P(<=0)=0.0022 | McNemar p=0.00901 (114/77)

## B. PLS-DA SNV dilatih ulang tanpa telur bertanda (LODO)
PLS-DA SNV bersih: n=1171 bal_acc=0.847 [0.826,0.867] AUC=0.922

## C. Model kubus dilatih ulang tanpa telur bertanda
yolo11s_all16: 3 seed, n=1171, bal_acc per seed 0.829 ± 0.004
  rata-rata seed: bal_acc=0.835 [0.814,0.856] AUC=0.920
specpool_all16: 3 seed, n=1171, bal_acc per seed 0.793 ± 0.003
  rata-rata seed: bal_acc=0.795 [0.771,0.818] AUC=0.872
Spasial vs non-spasial: dAUC=+0.0478 [+0.0313,+0.0642] P(<=0)=0.0000 | dBalAcc=+0.0407 [+0.0172,+0.0647] P(<=0)=0.0004 | McNemar p=0.00106 (123/76)
PLS bersih + yolo11s_all16: bal_acc=0.865 [0.845,0.885] AUC=0.938
  vs PLS bersih: dAUC=+0.0160 [+0.0085,+0.0239] P(<=0)=0.0000 | dBalAcc=+0.0182 [+0.0008,+0.0358] P(<=0)=0.0216 | McNemar p=0.0572 (66/45)
PLS bersih + specpool_all16: bal_acc=0.848 [0.827,0.868] AUC=0.913
  vs PLS bersih: dAUC=-0.0083 [-0.0157,-0.0008] P(<=0)=0.9840 | dBalAcc=+0.0007 [-0.0170,+0.0184] P(<=0)=0.4660 | McNemar p=1 (56/55)