# Cek coretan/tinta di cangkang

Parameter: band 549-700 nm, top-hat 5 px, >= 15% lebih gelap, komponen >= 15 px, ambang 'bertinta' skor > 0.002

fertil  : bertinta 13/612 (2.1%), skor median 0.0000, p90 0.0004
infertil: bertinta 21/616 (3.4%), skor median 0.0000, p90 0.0006
Fisher exact (bertinta vs kelas): p=0.223
Mann-Whitney skor tinta F vs IF: p=0.0309
AUC skor tinta untuk menebak 'fertil': 0.478 (0,5 = tidak informatif)

Proporsi bertinta per tanggal:
y        IF     F
date             
02Dec  0.08  0.04
03Dec  0.04  0.00
03Feb  0.07  0.00
04Dec  0.08  0.05
05Dec  0.08  0.04
06Dec  0.04  0.04
09Dec  0.08  0.04
10Dec  0.00  0.04
11Dec  0.04  0.00
12Dec  0.00  0.04
16Dec  0.00  0.00
17Dec  0.00  0.00
18Dec  0.00  0.00
19Dec  0.00  0.00
22Jan  0.04  0.00
23Jan  0.00  0.00
24Jan  0.00  0.00
25Nov   NaN  0.10
26Nov  0.04  0.06
27Jan  0.00  0.04
27Nov  0.04  0.00
28Jan  0.00  0.00
29Jan  0.00  0.04
30Jan  0.04  0.00
31Jan  0.08  0.00

Akurasi model pada telur bertinta vs tidak (prediksi OOF):
  pls              bertinta     n=  34 acc=0.794
  pls              tanpa tinta  n=1194 acc=0.857
  yolo11s_all16    bertinta     n=  34 acc=0.824
  yolo11s_all16    tanpa tinta  n=1194 acc=0.846
  specpool_all16   bertinta     n=  34 acc=0.735
  specpool_all16   tanpa tinta  n=1194 acc=0.816