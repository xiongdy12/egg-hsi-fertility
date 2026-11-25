# Eksperimen tambahan (NCA)

n=1228; seed YOLO=5, kontrol=5, anchor=5, anchor-kontrol=3

## A. Backbone dengan front end all16 yang sama (seed 0) + HybridSN 3D-2D
YOLO11s (all16)                    bal_acc=0.844 [0.822,0.863] AUC=0.922 MCC=0.687
YOLO11n                            bal_acc=0.835 [0.814,0.856] AUC=0.907 MCC=0.670 | McNemar vs YOLO11s p=0.343 (50/61)
YOLO11m                            bal_acc=0.840 [0.819,0.860] AUC=0.917 MCC=0.679 | McNemar vs YOLO11s p=0.691 (48/53)
ResNet-18                          bal_acc=0.833 [0.813,0.854] AUC=0.914 MCC=0.666 | McNemar vs YOLO11s p=0.319 (66/79)
MobileNetV3-small                  bal_acc=0.854 [0.835,0.875] AUC=0.928 MCC=0.709 | McNemar vs YOLO11s p=0.309 (76/63)
EfficientNet-B0                    bal_acc=0.841 [0.820,0.862] AUC=0.921 MCC=0.683 | McNemar vs YOLO11s p=0.872 (76/79)
HybridSN 3D-2D (75 bin)            bal_acc=0.824 [0.802,0.844] AUC=0.897 MCC=0.648 | McNemar vs YOLO11s p=0.109 (91/115)

## B. Ablasi lebar proyeksi k (seed 0): spasial (YOLO11s) vs non-spasial (kontrol)
k= 4  YOLO11s 0.835/0.907  kontrol 0.810/0.873  | dAUC=+0.0343 [+0.0158,+0.0523] P(<=0)=0.0000 | dBalAcc=+0.0245 [+0.0008,+0.0480] P(<=0)=0.0216 | McNemar p=0.0472 (122/92)
k= 8  YOLO11s 0.845/0.919  kontrol 0.796/0.873  | dAUC=+0.0469 [+0.0303,+0.0636] P(<=0)=0.0000 | dBalAcc=+0.0490 [+0.0262,+0.0720] P(<=0)=0.0000 | McNemar p=4.18e-05 (135/75)
k=16  YOLO11s 0.844/0.922  kontrol 0.814/0.878  | dAUC=+0.0443 [+0.0278,+0.0616] P(<=0)=0.0000 | dBalAcc=+0.0302 [+0.0068,+0.0534] P(<=0)=0.0058 | McNemar p=0.0134 (125/88)
k=32  YOLO11s 0.840/0.919  kontrol 0.814/0.889  | dAUC=+0.0303 [+0.0151,+0.0457] P(<=0)=0.0000 | dBalAcc=+0.0263 [+0.0047,+0.0482] P(<=0)=0.0080 | McNemar p=0.0273 (115/83)
k=64  YOLO11s 0.833/0.912  kontrol 0.817/0.896  | dAUC=+0.0156 [+0.0009,+0.0295] P(<=0)=0.0198 | dBalAcc=+0.0163 [-0.0044,+0.0369] P(<=0)=0.0636 | McNemar p=0.145 (95/75)

## C. Model usulan: YOLO11s ber-anchor PLS-DA (residual), 5 seed
per seed (n=5): bal_acc 0.847 ± 0.006 | AUC 0.915 ± 0.002
Anchor-YOLO11s (ensemble)          bal_acc=0.857 [0.836,0.876] AUC=0.922 MCC=0.714
  vs PLS-DA: dAUC=-0.0072 [-0.0120,-0.0025] P(<=0)=0.9982 | dBalAcc=+0.0016 [-0.0112,+0.0145] P(<=0)=0.4120 | McNemar p=0.901 (33/31)
  vs YOLO11s ensemble: dAUC=-0.0043 [-0.0175,+0.0089] P(<=0)=0.7414 | dBalAcc=+0.0114 [-0.0096,+0.0316] P(<=0)=0.1394 | McNemar p=0.31 (89/75)
  vs fusi rata-rata: dAUC=-0.0220 [-0.0297,-0.0145] P(<=0)=1.0000 | dBalAcc=-0.0213 [-0.0365,-0.0057] P(<=0)=0.9974 | McNemar p=0.00878 (33/59)
Anchor-kontrol (3 seed)            bal_acc=0.871 [0.852,0.889] AUC=0.940 MCC=0.743
  vs PLS-DA: dAUC=+0.0105 [+0.0014,+0.0199] P(<=0)=0.0136 | dBalAcc=+0.0161 [-0.0018,+0.0342] P(<=0)=0.0410 | McNemar p=0.0953 (75/55)
  Anchor-YOLO vs Anchor-kontrol (nilai tambah spasial di atas PLS): dAUC=-0.0177 [-0.0280,-0.0077] P(<=0)=0.9998 | dBalAcc=-0.0145 [-0.0333,+0.0039] P(<=0)=0.9382 | McNemar p=0.139 (57/75)

## D. Fusi dengan SATU model YOLO11s (tanpa ensemble)
fusi PLS + 1 seed (n=5): bal_acc 0.869 ± 0.004 (min 0.865) | AUC 0.941 ± 0.004
anchor 1 seed: bal_acc 0.847 ± 0.006 (min 0.838)

## E. Kalibrasi dan titik operasi (fertil = positif; % infertil tersortir saat X% fertil dipertahankan)
PLS-DA                 Brier=0.110 ECE=0.062 | infertil tersortir @fertil95%=0.734 @fertil99%=0.420
YOLO11s ens            Brier=0.110 ECE=0.027 | infertil tersortir @fertil95%=0.698 @fertil99%=0.537
Kontrol ens            Brier=0.136 ECE=0.050 | infertil tersortir @fertil95%=0.589 @fertil99%=0.326
Fusi                   Brier=0.097 ECE=0.061 | infertil tersortir @fertil95%=0.760 @fertil99%=0.554
Anchor-YOLO11s ens     Brier=0.108 ECE=0.035 | infertil tersortir @fertil95%=0.696 @fertil99%=0.383

## F. Uji temporal maju (latih s.d. 18Dec, uji sesudahnya)
latih 719 telur / 15 tanggal; uji 509 telur / 10 tanggal; PLS {'m__n_components': 15}
PLS-DA                             bal_acc=0.652 [0.617,0.684] AUC=0.757 MCC=0.372
YOLO11s (5 seed)                   bal_acc=0.670 [0.636,0.704] AUC=0.808 MCC=0.406 | McNemar vs PLS-DA p=0.272 (31/22)
Kontrol (3 seed)                   bal_acc=0.646 [0.617,0.675] AUC=0.761 MCC=0.396 | McNemar vs PLS-DA p=0.78 (24/27)
Fusi PLS+YOLO11s                   bal_acc=0.666 [0.634,0.699] AUC=0.794 MCC=0.407 | McNemar vs PLS-DA p=0.349 (24/17)
Anchor-YOLO11s (5 seed)            bal_acc=0.656 [0.623,0.689] AUC=0.794 MCC=0.384 | McNemar vs PLS-DA p=0.832 (12/10)
  Fusi vs PLS: dAUC=+0.0369 [+0.0246,+0.0500] P(<=0)=0.0000 | dBalAcc=+0.0140 [-0.0104,+0.0392] P(<=0)=0.1374 | McNemar p=0.349 (24/17)
  Anchor vs PLS: dAUC=+0.0371 [+0.0259,+0.0484] P(<=0)=0.0000 | dBalAcc=+0.0040 [-0.0139,+0.0219] P(<=0)=0.3400 | McNemar p=0.832 (12/10)
  YOLO vs kontrol: dAUC=+0.0469 [+0.0258,+0.0688] P(<=0)=0.0000 | dBalAcc=+0.0230 [+0.0007,+0.0451] P(<=0)=0.0214 | McNemar p=0.0652 (24/12)