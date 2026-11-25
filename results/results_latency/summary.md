# Benchmark efisiensi

Protokol: {"cpu": "INTEL(R) XEON(R) PLATINUM 8570", "torch": "2.11.0+cu128", "res": 128, "bands": 300, "threads": [1, 4], "reps": 300, "warmup": 50, "batch": 1, "dtype": "fp32", "mode": "eager", "gpu": "NVIDIA B200"}

                             tag   k  params_M  GFLOPs_sensor  GFLOPs_hsi  cpu1_sensor_ms  cpu1_hsi_ms  cpu4_sensor_ms  cpu4_hsi_ms  gpu_hsi_ms
             yolo11s_concrete_k3   3     5.446          0.479       0.509           6.006        6.867           3.822        4.132       2.193
                  yolo11s_led_k3   3     5.446          0.479       0.509           6.114        7.224           3.912        4.310       2.261
            specpool_concrete_k3   3     0.001          0.000       0.029           0.096        0.927           0.120        0.387       0.363
             yolo11s_concrete_k5   5     5.448          0.484       0.533           6.006        6.912           3.883        4.211       2.226
                  yolo11s_led_k5   5     5.446          0.484       0.533           6.067        7.975           3.887        4.457       2.280
            specpool_concrete_k5   5     0.002          0.000       0.049           0.116        0.938           0.118        0.395       0.364
             yolo11s_concrete_k8   8     5.449          0.491       0.570           6.157        7.058           3.849        4.155       2.219
                  yolo11s_led_k8   8     5.447          0.491       0.570           6.131        9.338           3.834        4.724       2.328
            specpool_concrete_k8   8     0.003          0.000       0.079           0.166        1.069           0.105        0.412       0.371
                     yolo11s_rgb   3     5.446          0.479       0.509           6.151        6.392           3.860        3.974       2.169
                    yolo11s_vip5   5     5.446          0.484       0.533           5.886        6.619           3.864        4.123       2.149
                   yolo11s_all16  16     5.454          0.667       0.667          49.228       49.344          18.991       18.900       2.119
                  specpool_all16  16     0.006          0.157       0.157          44.208       44.207          15.695       15.566       0.355
             yolo11n_concrete_k5   5     1.535          0.129       0.178           2.676        3.446           2.426        2.754       2.172
             yolo11m_concrete_k5   5    10.358          1.580       1.629          13.870       14.705           7.260        7.563       2.645
  mobilenet_v3_small_concrete_k5   5     1.522          0.040       0.089           2.222        3.063           2.282        2.589       2.115
     efficientnet_b0_concrete_k5   5     4.012          0.257       0.306           5.945        6.814           5.162        5.541       2.971
            resnet18_concrete_k5   5    11.185          1.236       1.285          10.124       10.961           4.207        4.533       1.109
PLS-DA (SNV, spektrum rata-rata) 300     0.000          0.005       0.005          13.536       13.536           3.666        3.666         NaN