# Analisis akhir

## Q1. Stabilitas antar-seed
yolo11s_all16: 5 seed, n=1228 | bal_acc 0.834 ± 0.011 (min 0.818, maks 0.844) | AUC 0.917 ± 0.006
  rata-rata 5 seed: bal_acc=0.845 [0.825,0.865] AUC=0.927 MCC=0.691
specpool_all16: 5 seed, n=1228 | bal_acc 0.814 ± 0.005 (min 0.809, maks 0.821) | AUC 0.881 ± 0.003
  rata-rata 5 seed: bal_acc=0.814 [0.792,0.835] AUC=0.884 MCC=0.627

Pembanding PLS-DA SNV (CSV): bal_acc=0.855 AUC=0.930 (n=1228)

## Q2. Spasial vs non-spasial: yolo11s_all16 vs specpool_all16 (rata-rata seed)
dAUC=+0.0426 [+0.0278,+0.0583] P(<=0)=0.0000 | dBalAcc=+0.0318 [+0.0096,+0.0543] P(<=0)=0.0030 | McNemar p=0.00609 (116/77)

## Q3. Fusi dengan PLS-DA (rata-rata sederhana, a priori)
PLS + yolo11s_all16: bal_acc=0.878 [0.859,0.895] AUC=0.944 MCC=0.756
  vs PLS saja: dAUC=+0.0148 [+0.0076,+0.0224] P(<=0)=0.0000 | dBalAcc=+0.0228 [+0.0052,+0.0406] P(<=0)=0.0050 | McNemar p=0.015 (76/48)
PLS + specpool_all16: bal_acc=0.856 [0.837,0.875] AUC=0.923 MCC=0.712
  vs PLS saja: dAUC=-0.0069 [-0.0139,+0.0002] P(<=0)=0.9706 | dBalAcc=+0.0006 [-0.0171,+0.0179] P(<=0)=0.4750 | McNemar p=1 (60/59)