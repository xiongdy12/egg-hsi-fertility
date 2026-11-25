# Uji spasial YOLOv11s (25 fold, n=1228)

asli                             bal_acc=0.835 AUC=0.919
piksel diacak                    bal_acc=0.502 AUC=0.761
diratakan (spektrum rata-rata)   bal_acc=0.675 AUC=0.731
asli - acak    dAUC=+0.1579 [+0.1317,+0.1846]  dBalAcc=+0.3339 [+0.3126,+0.3545]
asli - rata    dAUC=+0.1877 [+0.1604,+0.2166]  dBalAcc=+0.1608 [+0.1295,+0.1921]

Profil radial Grad-CAM (telur terklasifikasi benar; 0 = pusat, 1 = tepi):
  fertil   0.55 0.54 0.49 0.42 0.35 0.28 0.22 0.17 0.13 0.10
  infertil 0.36 0.36 0.36 0.35 0.34 0.31 0.28 0.23 0.18 0.12