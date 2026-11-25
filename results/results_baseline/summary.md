# Baseline spektral (CSV)

Data: 1228 telur, 300 band (374.1-1015.3 nm), fertil=612, infertil=616, tanggal=25
Cek label ID vs kolom 'Fertility status': {'Fertile': {'F': 612, 'IF': 0}, 'Infertile': {'F': 0, 'IF': 616}}
Tanggal dengan satu kelas saja: ['25Nov']

## A. Reproduksi protokol paper (split acak stratified 70/15/15, PLS-DA SG2), 10 ulangan
Akurasi test acak: 0.909 ± 0.015 (paper: 0.903)

## B. LODO CV, spektrum penuh
raw  PLS-DA   bal_acc=0.757 [0.732,0.781] auc=0.832 (9s)
raw  LogReg   bal_acc=0.772 [0.748,0.796] auc=0.847 (9s)
raw  SVM-RBF  bal_acc=0.779 [0.755,0.804] auc=0.862 (29s)
raw  HGB      bal_acc=0.731 [0.706,0.756] auc=0.807 (21s)
raw  RF       bal_acc=0.700 [0.674,0.726] auc=0.784 (51s)
SNV  PLS-DA   bal_acc=0.855 [0.836,0.875] auc=0.930 (8s)
SNV  LogReg   bal_acc=0.844 [0.822,0.864] auc=0.908 (9s)
SNV  SVM-RBF  bal_acc=0.818 [0.798,0.839] auc=0.887 (24s)
SNV  HGB      bal_acc=0.768 [0.744,0.792] auc=0.817 (21s)
SNV  RF       bal_acc=0.777 [0.754,0.800] auc=0.839 (47s)
MSC  PLS-DA   bal_acc=0.847 [0.827,0.866] auc=0.919 (8s)
MSC  LogReg   bal_acc=0.831 [0.809,0.853] auc=0.906 (9s)
MSC  SVM-RBF  bal_acc=0.821 [0.801,0.842] auc=0.888 (25s)
MSC  HGB      bal_acc=0.766 [0.741,0.789] auc=0.818 (22s)
MSC  RF       bal_acc=0.775 [0.752,0.799] auc=0.839 (49s)
SG1  PLS-DA   bal_acc=0.811 [0.789,0.832] auc=0.881 (8s)
SG1  LogReg   bal_acc=0.831 [0.811,0.852] auc=0.893 (9s)
SG1  SVM-RBF  bal_acc=0.802 [0.779,0.824] auc=0.871 (24s)
SG1  HGB      bal_acc=0.775 [0.752,0.800] auc=0.854 (24s)
SG1  RF       bal_acc=0.775 [0.750,0.799] auc=0.848 (47s)
SG2  PLS-DA   bal_acc=0.792 [0.769,0.814] auc=0.855 (8s)
SG2  LogReg   bal_acc=0.810 [0.789,0.832] auc=0.868 (9s)
SG2  SVM-RBF  bal_acc=0.833 [0.810,0.853] auc=0.899 (25s)
SG2  HGB      bal_acc=0.822 [0.799,0.842] auc=0.890 (24s)
SG2  RF       bal_acc=0.804 [0.782,0.827] auc=0.887 (49s)
SNV  1D-CNN   bal_acc=0.735 [0.712,0.759] auc=0.779
SG1  1D-CNN   bal_acc=0.742 [0.717,0.765] auc=0.805

## C. Seleksi band di dalam fold (k band) dan RGB-simulasi
Preprocessing terbaik dari blok B: SNV (dipakai untuk seleksi band)
VIP    k= 3 LogReg   bal_acc=0.749 [0.724,0.773] band~[510.0, 659.0, 672.0]
VIP    k= 3 SVM-RBF  bal_acc=0.741 [0.715,0.764] band~[510.0, 659.0, 672.0]
VIP    k= 5 LogReg   bal_acc=0.756 [0.731,0.779] band~[499.0, 510.0, 659.0, 672.0, 745.0]
VIP    k= 5 SVM-RBF  bal_acc=0.768 [0.744,0.792] band~[499.0, 510.0, 659.0, 672.0, 745.0]
VIP    k= 8 LogReg   bal_acc=0.757 [0.734,0.781] band~[499.0, 510.0, 520.0, 623.0, 659.0, 672.0, 745.0, 763.0]
VIP    k= 8 SVM-RBF  bal_acc=0.783 [0.760,0.806] band~[499.0, 510.0, 520.0, 623.0, 659.0, 672.0, 745.0, 763.0]
VIP    k=10 LogReg   bal_acc=0.755 [0.731,0.778] band~[499.0, 510.0, 520.0, 623.0, 649.0, 659.0, 672.0, 683.0, 745.0, 763.0]
VIP    k=10 SVM-RBF  bal_acc=0.780 [0.758,0.802] band~[499.0, 510.0, 520.0, 623.0, 649.0, 659.0, 672.0, 683.0, 745.0, 763.0]
SPA    k= 3 LogReg   bal_acc=0.719 [0.693,0.745] band~[510.0, 594.0, 644.0]
SPA    k= 3 SVM-RBF  bal_acc=0.721 [0.696,0.747] band~[510.0, 594.0, 644.0]
SPA    k= 5 LogReg   bal_acc=0.753 [0.728,0.777] band~[510.0, 594.0, 634.0, 644.0, 674.0]
SPA    k= 5 SVM-RBF  bal_acc=0.750 [0.725,0.774] band~[510.0, 594.0, 634.0, 644.0, 674.0]
SPA    k= 8 LogReg   bal_acc=0.756 [0.731,0.780] band~[510.0, 556.0, 594.0, 615.0, 634.0, 644.0, 674.0, 750.0]
SPA    k= 8 SVM-RBF  bal_acc=0.784 [0.761,0.806] band~[510.0, 556.0, 594.0, 615.0, 634.0, 644.0, 674.0, 750.0]
SPA    k=10 LogReg   bal_acc=0.742 [0.716,0.766] band~[510.0, 556.0, 579.0, 594.0, 615.0, 634.0, 644.0, 674.0, 702.0, 750.0]
SPA    k=10 SVM-RBF  bal_acc=0.777 [0.753,0.800] band~[510.0, 556.0, 579.0, 594.0, 615.0, 634.0, 644.0, 674.0, 702.0, 750.0]
Fisher k= 3 LogReg   bal_acc=0.739 [0.714,0.764] band~[510.0, 520.0, 672.0]
Fisher k= 3 SVM-RBF  bal_acc=0.739 [0.714,0.764] band~[510.0, 520.0, 672.0]
Fisher k= 5 LogReg   bal_acc=0.763 [0.739,0.786] band~[499.0, 510.0, 520.0, 659.0, 672.0]
Fisher k= 5 SVM-RBF  bal_acc=0.782 [0.759,0.805] band~[499.0, 510.0, 520.0, 659.0, 672.0]
Fisher k= 8 LogReg   bal_acc=0.765 [0.741,0.788] band~[489.0, 499.0, 510.0, 520.0, 659.0, 672.0, 802.0, 813.0]
Fisher k= 8 SVM-RBF  bal_acc=0.760 [0.735,0.783] band~[489.0, 499.0, 510.0, 520.0, 659.0, 672.0, 802.0, 813.0]
Fisher k=10 LogReg   bal_acc=0.757 [0.733,0.781] band~[489.0, 499.0, 510.0, 520.0, 659.0, 672.0, 791.0, 802.0, 813.0, 824.0]
Fisher k=10 SVM-RBF  bal_acc=0.772 [0.748,0.796] band~[489.0, 499.0, 510.0, 520.0, 659.0, 672.0, 791.0, 802.0, 813.0, 824.0]
RGB-sim     LogReg   bal_acc=0.599 [0.572,0.627]
RGB-sim     SVM-RBF  bal_acc=0.608 [0.583,0.635]

## D. Uji confound: morfometri saja (tanpa spektrum)
Mass (g)             F=59.11±3.99  IF=61.16±4.52  Mann-Whitney p=4.66e-16
Major dia (mm)       F=57.59±1.84  IF=57.82±2.18  Mann-Whitney p=3.56e-02
Minor dia (mm)       F=42.98±1.05  IF=43.56±1.18  Mann-Whitney p=3.30e-17
Thickness (mm)       F=0.36±0.03  IF=0.35±0.03  Mann-Whitney p=8.86e-19
Shell strength (N)   F=29.40±6.40  IF=27.81±6.50  Mann-Whitney p=1.96e-05
Morfometri LogReg   bal_acc=0.668 auc=0.718 (~0.5 = tidak ada confound; tinggi = perbedaan flok terdeteksi tanpa spektrum)
Morfometri HGB      bal_acc=0.661 auc=0.693 (~0.5 = tidak ada confound; tinggi = perbedaan flok terdeteksi tanpa spektrum)

## E. Uji statistik vs baseline PLS-DA SG2 (protokol paper, di bawah LODO)
B|SNV|PLS-DA                 McNemar p=5.53e-09 (benar-salah 129/51); Wilcoxon per-tanggal p=0.0157
B|MSC|PLS-DA                 McNemar p=1.92e-07 (benar-salah 119/51); Wilcoxon per-tanggal p=0.0307
B|SNV|LogReg                 McNemar p=1.44e-05 (benar-salah 139/75); Wilcoxon per-tanggal p=0.0679
B|SG2|SVM-RBF                McNemar p=0.000546 (benar-salah 131/80); Wilcoxon per-tanggal p=0.174
B|SG1|LogReg                 McNemar p=0.000285 (benar-salah 113/64); Wilcoxon per-tanggal p=0.163
B|MSC|LogReg                 McNemar p=0.000961 (benar-salah 131/82); Wilcoxon per-tanggal p=0.125
B|SG2|HGB                    McNemar p=0.0316 (benar-salah 159/122); Wilcoxon per-tanggal p=0.306
B|MSC|SVM-RBF                McNemar p=0.0383 (benar-salah 161/125); Wilcoxon per-tanggal p=0.498

Selesai dalam 18.0 menit. Hasil: results_baseline/

## 10 teratas (LODO)
 block prep   model bands  bal_acc  bal_acc_lo  bal_acc_hi      auc      mcc
B_full  SNV  PLS-DA   NaN 0.855106    0.835727    0.874537 0.929544 0.710585
B_full  MSC  PLS-DA   NaN 0.846941    0.826884    0.865844 0.918795 0.694011
B_full  SNV  LogReg   NaN 0.843684    0.822465    0.864408 0.908165 0.687495
B_full  SG2 SVM-RBF   NaN 0.833105    0.810425    0.852875 0.898828 0.666406
B_full  SG1  LogReg   NaN 0.831498    0.810504    0.851869 0.892674 0.663458
B_full  MSC  LogReg   NaN 0.831455    0.809496    0.852659 0.906486 0.662950
B_full  SG2     HGB   NaN 0.821757    0.799231    0.842436 0.890343 0.644560
B_full  MSC SVM-RBF   NaN 0.820925    0.800743    0.841646 0.888331 0.642514
B_full  SNV SVM-RBF   NaN 0.818474    0.797814    0.839440 0.887449 0.637472
B_full  SG1  PLS-DA   NaN 0.811004    0.789455    0.832237 0.880533 0.622652