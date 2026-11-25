# Analisis confound flok

n=1228 (baris tanpa morfometri dibuang: 0), morfometri: ['Mass (g)', 'Major dia (mm)', 'Minor dia (mm)', 'Thickness (mm)', 'Shell strength (N)']

## 1-4. LODO: spektrum vs morfometri vs gabungan vs spektrum-residual
PLS-DA  spec   bal_acc=0.855 [0.836,0.875] auc=0.930
PLS-DA  morph  bal_acc=0.644 [0.617,0.671] auc=0.700
PLS-DA  both   bal_acc=0.870 [0.851,0.888] auc=0.935
PLS-DA  resid  bal_acc=0.797 [0.775,0.819] auc=0.867
  -> nilai tambah spektrum di atas morfometri (PLS-DA): dAUC=+0.236 [+0.206,+0.266], P(dAUC<=0)=0.0000
  -> efek residualisasi (PLS-DA): dAUC(resid-spec)=-0.063 [-0.077,-0.049]
LogReg  spec   bal_acc=0.844 [0.822,0.864] auc=0.908
LogReg  morph  bal_acc=0.668 [0.641,0.695] auc=0.718
LogReg  both   bal_acc=0.827 [0.806,0.848] auc=0.906
LogReg  resid  bal_acc=0.788 [0.766,0.811] auc=0.862
  -> nilai tambah spektrum di atas morfometri (LogReg): dAUC=+0.188 [+0.161,+0.217], P(dAUC<=0)=0.0000
  -> efek residualisasi (LogReg): dAUC(resid-spec)=-0.046 [-0.060,-0.033]

## 5. Subset matched per tanggal (morfometri seimbang)

Caliper 1.0 SD: 384 telur (192 F / 192 IF), 24 tanggal
  SMD Mass (g)             sebelum=-0.48  sesudah=-0.13
  SMD Major dia (mm)       sebelum=-0.11  sesudah=+0.07
  SMD Minor dia (mm)       sebelum=-0.51  sesudah=+0.01
  SMD Thickness (mm)       sebelum=+0.54  sesudah=+0.04
  SMD Shell strength (N)   sebelum=+0.25  sesudah=+0.05
  PLS-DA  spec   bal_acc=0.846 [0.810,0.881] auc=0.923
  PLS-DA  morph  bal_acc=0.562 [0.514,0.610] auc=0.586
  LogReg  spec   bal_acc=0.828 [0.790,0.866] auc=0.900
  LogReg  morph  bal_acc=0.604 [0.554,0.651] auc=0.635

Caliper 1.5 SD: 614 telur (307 F / 307 IF), 24 tanggal
  SMD Mass (g)             sebelum=-0.48  sesudah=-0.13
  SMD Major dia (mm)       sebelum=-0.11  sesudah=+0.08
  SMD Minor dia (mm)       sebelum=-0.51  sesudah=-0.03
  SMD Thickness (mm)       sebelum=+0.54  sesudah=+0.12
  SMD Shell strength (N)   sebelum=+0.25  sesudah=+0.03
  PLS-DA  spec   bal_acc=0.853 [0.824,0.882] auc=0.927
  PLS-DA  morph  bal_acc=0.557 [0.517,0.596] auc=0.573
  LogReg  spec   bal_acc=0.826 [0.794,0.855] auc=0.901
  LogReg  morph  bal_acc=0.572 [0.531,0.610] auc=0.616

Selesai dalam 2.3 menit.

Cara membaca: kalau (a) 'both' > 'morph' secara signifikan, (b) 'resid' tidak jauh di bawah 'spec', dan (c) di subset matched 'morph' ~0.5 sementara 'spec' tetap tinggi -> sinyal spektral tidak dijelaskan oleh morfometri flok.