固定 dev400 前两句：reference 与解码输出

句子按数据文件原顺序固定，不按好坏筛选。REF/HYP 都使用本次评分的同一 IPA→ARPAbet 映射。
对齐中 * 表示插入/删除处的空位；每个完整 topology 音素 occurrence 保留，未事后合并重复音素。

jsd_fs / final / timit:fadg0_si1279

参考 18 音素，预测 61 音素；S=2 D=0 I=43，PER=250.00%

```text
REF: b r ih k s aa r er n aa l t er n ih t ih v
HYP: b b b b b b r ih ih k k k k s s s aa aa er aa er er er er r ae ae n n aa aa aa l l t t t t t t er er er UNMAPPED_IPA:ɲ ah ih t t t t t ih ih ih ih v v v v v v
GREEDY: b b b b ih ih k k k s s s er er er er ae n n n aa aa l l t t t t er er er UNMAPPED_IPA:ɲ ih t t t ih ih ih v v v v v
```

编辑对齐：

```text
REF: b  *  *  *  *  *  r  ih *  k  *  *  *  s  *  *  aa *  *  *  *  *  *  *  r 
HYP: b  b  b  b  b  b  r  ih ih k  k  k  k  s  s  s  aa aa er aa er er er er r 
REF: *  er n  *  aa *  *  l  *  t  *  *  *  *  *  er *  *  *              n  ih t  *  *  * 
HYP: ae ae n  n  aa aa aa l  l  t  t  t  t  t  t  er er er UNMAPPED_IPA:ɲ ah ih t  t  t  t 
REF: *  ih *  *  *  *  *  *  *  *  v 
HYP: t  ih ih ih ih v  v  v  v  v  v 
```

前40段合法状态路径（每帧10ms）：

```text
   0–  10 ms  b:S0
  10–  20 ms  b:S2
  20–  30 ms  blank
  30–  50 ms  b:S0
  50–  60 ms  b:S2
  60–  70 ms  blank
  70–  90 ms  b:S0
  90– 100 ms  b:S2
 100– 110 ms  b:S0
 110– 120 ms  b:S2
 120– 130 ms  bʲ:S0
 130– 140 ms  bʲ:S2
 140– 150 ms  b:S0
 150– 160 ms  b:S2
 160– 170 ms  ɹ:S0
 170– 190 ms  ɹ:S2
 190– 270 ms  ɪ:S0
 270– 280 ms  ɪ:S2
 280– 310 ms  ɪ:S0
 310– 320 ms  ɪ:S2
 320– 330 ms  blank
 330– 370 ms  k:S0
 370– 380 ms  k:S2
 380– 390 ms  k:S0
 390– 400 ms  k:S2
 400– 410 ms  k:S0
 410– 420 ms  k:S2
 420– 430 ms  k:S0
 430– 440 ms  k:S2
 440– 450 ms  s:S0
 450– 460 ms  s:S2
 460– 490 ms  s:S0
 490– 500 ms  s:S2
 500– 510 ms  s:S0
 510– 520 ms  s:S2
 520– 540 ms  blank
 540– 570 ms  ɒ:S0
 570– 580 ms  ɒ:S2
 580– 590 ms  ɒ:S0
 590– 600 ms  ɒ:S2
```

jsd_fs / final / timit:fadg0_si1909

参考 24 音素，预测 94 音素；S=2 D=0 I=70，PER=300.00%

```text
REF: f ae t sh ow d ih n l uw s r ow l z b ih n iy th ih sh er t
HYP: f f f f f f ae ae ae ae ae ae ae t t t t t sh sh sh sh sh sh sh ow ow ow d d d d ae ih ih n l UNMAPPED_IPA:ʎ UNMAPPED_IPA:ʎ uw uw uw uw uw s th s s s r r ow l l l l l z z s b b b b ih UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ uw iy t th th th d d d d ah ah sh sh sh sh sh er er er er er er er t t t
GREEDY: f f f f f ae ae ae ae ae ae t t t sh sh sh sh ow d d ae n l uw uw uw uw uw s s r ow l l l z b b UNMAPPED_IPA:ɲ uw t th th d d ah ah sh sh sh sh er er er er er er t t t
```

编辑对齐：

```text
REF: f  *  *  *  *  *  ae *  *  *  *  *  *  t  *  *  *  *  sh *  *  *  *  *  * 
HYP: f  f  f  f  f  f  ae ae ae ae ae ae ae t  t  t  t  t  sh sh sh sh sh sh sh
REF: ow *  *  d  *  *  *  *  ih *  n  l  *              *              uw *  *  *  *  s  *  *  *  *  r 
HYP: ow ow ow d  d  d  d  ae ih ih n  l  UNMAPPED_IPA:ʎ UNMAPPED_IPA:ʎ uw uw uw uw uw s  th s  s  s  r 
REF: *  ow l  *  *  *  *  z  *  *  b  *  *  *  ih *              *              n  iy *  th *  *  *  * 
HYP: r  ow l  l  l  l  l  z  z  s  b  b  b  b  ih UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ uw iy t  th th th d  d 
REF: *  *  *  ih sh *  *  *  *  er *  *  *  *  *  *  *  *  t 
HYP: d  d  ah ah sh sh sh sh sh er er er er er er er t  t  t 
```

前40段合法状态路径（每帧10ms）：

```text
   0–  30 ms  blank
  30–  50 ms  f:S0
  50–  60 ms  f:S2
  60–  70 ms  f:S0
  70–  80 ms  f:S2
  80–  90 ms  f:S0
  90– 100 ms  f:S2
 100– 110 ms  f:S0
 110– 120 ms  f:S2
 120– 130 ms  f:S0
 130– 140 ms  f:S2
 140– 150 ms  f:S0
 150– 160 ms  f:S2
 160– 180 ms  blank
 180– 270 ms  æ:S0
 270– 280 ms  æ:S2
 280– 310 ms  æ:S0
 310– 320 ms  æ:S2
 320– 330 ms  æ:S0
 330– 340 ms  æ:S2
 340– 350 ms  æ:S0
 350– 360 ms  æ:S2
 360– 370 ms  æ:S0
 370– 380 ms  æ:S2
 380– 390 ms  æ:S0
 390– 400 ms  æ:S2
 400– 410 ms  æ:S0
 410– 420 ms  æ:S2
 420– 430 ms  blank
 430– 440 ms  t:S0
 440– 460 ms  t:S2
 460– 490 ms  t:S0
 490– 500 ms  t:S2
 500– 510 ms  t:S0
 510– 520 ms  t:S2
 520– 530 ms  t:S0
 530– 540 ms  t:S2
 540– 550 ms  t:S0
 550– 560 ms  t:S2
 560– 570 ms  ʃ:S0
```

jsd_fs / cumulative_average / timit:fadg0_si1279

参考 18 音素，预测 67 音素；S=3 D=0 I=49，PER=288.89%

```text
REF: b r ih k s aa r er n aa l t er n ih t ih v
HYP: y w f f w r ey ih k k s s s s s aa aa aa aa aa aa aa aa r r er ih ae n ae aa aa aa aa l l l th t t hh er er er er er m eh UNMAPPED_IPA:ɲ eh ih k ah t t t t t t t ih ih s ah t t z
GREEDY: w f r ey s s aa aa aa aa aa aa r n aa aa aa l l t er er er er eh k t t t t t ih z
```

编辑对齐：

```text
REF: *  *  *  *  b  r  *  ih k  *  s  *  *  *  *  aa *  *  *  *  *  *  *  r  * 
HYP: y  w  f  f  w  r  ey ih k  k  s  s  s  s  s  aa aa aa aa aa aa aa aa r  r 
REF: er *  *  n  *  aa *  *  *  l  *  *  *  t  *  *  er *  *  *  *  *  *  *              n 
HYP: er ih ae n  ae aa aa aa aa l  l  l  th t  t  hh er er er er er m  eh UNMAPPED_IPA:ɲ eh
REF: ih *  *  t  *  *  *  *  *  *  ih *  *  *  *  *  v 
HYP: ih k  ah t  t  t  t  t  t  t  ih ih s  ah t  t  z 
```

前40段合法状态路径（每帧10ms）：

```text
   0–  10 ms  j:S0
  10–  20 ms  j:S2
  20–  30 ms  blank
  30–  40 ms  w:S0
  40–  50 ms  w:S2
  50–  70 ms  blank
  70–  90 ms  f:S0
  90– 100 ms  f:S2
 100– 110 ms  blank
 110– 130 ms  f:S0
 130– 140 ms  f:S2
 140– 150 ms  blank
 150– 170 ms  w:S0
 170– 180 ms  w:S2
 180– 190 ms  ɹ:S0
 190– 250 ms  ɹ:S2
 250– 280 ms  ej:S0
 280– 290 ms  ej:S2
 290– 310 ms  ɪ:S0
 310– 320 ms  ɪ:S2
 320– 330 ms  k:S0
 330– 340 ms  k:S2
 340– 370 ms  k:S0
 370– 380 ms  k:S2
 380– 390 ms  blank
 390– 410 ms  s:S0
 410– 420 ms  s:S2
 420– 450 ms  s:S0
 450– 460 ms  s:S2
 460– 470 ms  s:S0
 470– 480 ms  s:S2
 480– 490 ms  s:S0
 490– 500 ms  s:S2
 500– 510 ms  s:S0
 510– 520 ms  s:S2
 520– 530 ms  ɒː:S0
 530– 540 ms  ɒː:S2
 540– 550 ms  ɒː:S0
 550– 560 ms  ɒː:S2
 560– 570 ms  ɒː:S0
```

jsd_fs / cumulative_average / timit:fadg0_si1909

参考 24 音素，预测 100 音素；S=12 D=0 I=76，PER=366.67%

```text
REF: f ae t sh ow d ih n l uw s r ow l z b ih n iy th ih sh er t
HYP: hh ae ae ae ae f f ae ae ae ae ae ae ae ae k ae ch sh sh sh ch jh er uw d ae d d ey ih ih ng ih ng ah l uw l uw uw uw uw uw z s s s f f r aa aa aa aa aa aa l l l l l s s s t ah ah ah d ah UNMAPPED_IPA:ɲ iy s s f f d ah ch ah ch ah ch t ch y er er er er er er er er er er er z z
GREEDY: ae ae ae ae ae ae ae ch sh sh sh d d ih l l uw uw uw z s s f r aa aa aa aa aa aa l l l l l s s t d s d ch ch ch y er er er er er er er er er er z
```

编辑对齐：

```text
REF: *  *  *  *  *  f  *  ae *  *  *  *  *  *  *  *  *  t  sh *  *  *  *  *  ow
HYP: hh ae ae ae ae f  f  ae ae ae ae ae ae ae ae k  ae ch sh sh sh ch jh er uw
REF: d  *  *  *  *  ih *  *  *  *  n  l  uw *  *  *  *  *  *  *  s  *  *  *  * 
HYP: d  ae d  d  ey ih ih ng ih ng ah l  uw l  uw uw uw uw uw z  s  s  s  f  f 
REF: r  *  *  *  *  *  ow l  *  *  *  *  *  *  *  *  *  *  z  b  ih n              iy *  * 
HYP: r  aa aa aa aa aa aa l  l  l  l  l  s  s  s  t  ah ah ah d  ah UNMAPPED_IPA:ɲ iy s  s 
REF: *  *  *  *  *  *  *  *  *  th ih sh er *  *  *  *  *  *  *  *  *  *  *  t 
HYP: f  f  d  ah ch ah ch ah ch t  ch y  er er er er er er er er er er er z  z 
```

前40段合法状态路径（每帧10ms）：

```text
   0–  10 ms  h:S0
  10–  20 ms  h:S2
  20–  30 ms  æ:S0
  30–  40 ms  æ:S2
  40–  60 ms  blank
  60–  70 ms  æ:S0
  70–  80 ms  æ:S2
  80– 100 ms  blank
 100– 110 ms  æ:S0
 110– 120 ms  æ:S2
 120– 140 ms  blank
 140– 150 ms  æ:S0
 150– 160 ms  æ:S2
 160– 190 ms  blank
 190– 210 ms  f:S0
 210– 220 ms  f:S2
 220– 230 ms  blank
 230– 250 ms  f:S0
 250– 260 ms  f:S2
 260– 270 ms  æ:S0
 270– 280 ms  æ:S2
 280– 300 ms  blank
 300– 310 ms  æ:S0
 310– 320 ms  æ:S2
 320– 330 ms  æ:S0
 330– 340 ms  æ:S2
 340– 350 ms  æ:S0
 350– 360 ms  æ:S2
 360– 370 ms  æ:S0
 370– 380 ms  æ:S2
 380– 390 ms  æ:S0
 390– 400 ms  æ:S2
 400– 410 ms  æ:S0
 410– 420 ms  æ:S2
 420– 430 ms  æ:S0
 430– 440 ms  æ:S2
 440– 450 ms  k:S0
 450– 460 ms  k:S2
 460– 470 ms  æ:S0
 470– 480 ms  æ:S2
```

jsd_only / final / timit:fadg0_si1279

参考 18 音素，预测 38 音素；S=2 D=0 I=20，PER=122.22%

```text
REF: b r ih k s aa r er n aa l t er n ih t ih v
HYP: UNMAPPED_IPA:ç b r b r r ih ih k s s aa aa aa r r ih n aa aa l l t t t er er er UNMAPPED_IPA:ɲ ih t ih v v v v v v
GREEDY: 
```

编辑对齐：

```text
REF: *              b  r  *  *  *  ih *  k  s  *  aa *  *  r  *  er n  aa *  l  *  t  *  * 
HYP: UNMAPPED_IPA:ç b  r  b  r  r  ih ih k  s  s  aa aa aa r  r  ih n  aa aa l  l  t  t  t 
REF: er *  *  n              ih t  ih *  *  *  *  *  v 
HYP: er er er UNMAPPED_IPA:ɲ ih t  ih v  v  v  v  v  v 
```

前40段合法状态路径（每帧10ms）：

```text
   0–  10 ms  blank
  10–  20 ms  ç:S0
  20–  30 ms  ç:S2
  30–  50 ms  b:S0
  50–  60 ms  b:S2
  60–  70 ms  blank
  70–  90 ms  ɹ:S0
  90– 120 ms  ɹ:S2
 120– 130 ms  b:S0
 130– 150 ms  b:S2
 150– 180 ms  ɹ:S0
 180– 190 ms  ɹ:S2
 190– 210 ms  ɹ:S0
 210– 250 ms  ɹ:S1
 250– 260 ms  ɹ:S2
 260– 310 ms  ɪ:S0
 310– 330 ms  ɪ:S1
 330– 340 ms  ɪ:S2
 340– 350 ms  ɪ:S0
 350– 370 ms  ɪ:S2
 370– 390 ms  k:S0
 390– 430 ms  k:S1
 430– 440 ms  k:S2
 440– 450 ms  blank
 450– 480 ms  s:S0
 480– 530 ms  s:S1
 530– 540 ms  s:S2
 540– 550 ms  blank
 550– 560 ms  s:S0
 560– 570 ms  s:S1
 570– 580 ms  s:S2
 580– 590 ms  blank
 590– 600 ms  ɒ:S0
 600– 610 ms  ɒ:S2
 610– 620 ms  ɑ:S0
 620– 630 ms  ɑ:S2
 630– 640 ms  ɒ:S0
 640– 670 ms  ɒ:S1
 670– 690 ms  ɒ:S2
 690– 700 ms  ɹ:S0
```

jsd_only / final / timit:fadg0_si1909

参考 24 音素，预测 58 音素；S=8 D=0 I=34，PER=175.00%

```text
REF: f ae t sh ow d ih n l uw s r ow l z b ih n iy th ih sh er t
HYP: f f f f f ae ae ae eh t t t aa sh n ow ow ow ow ow ow d d ih n UNMAPPED_IPA:ʎ uw uw s r r r ow ow iy l l l s l l eh UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ t t t d ah ah sh ih er er er er t
GREEDY: t
```

编辑对齐：

```text
REF: f  *  *  *  *  ae *  *  *  t  *  *  *  sh *  ow *  *  *  *  *  d  *  ih n 
HYP: f  f  f  f  f  ae ae ae eh t  t  t  aa sh n  ow ow ow ow ow ow d  d  ih n 
REF: l              uw *  s  r  *  *  ow *  *  l  *  *  *  *  *  *  *              *              *              *  z  b  ih n 
HYP: UNMAPPED_IPA:ʎ uw uw s  r  r  r  ow ow iy l  l  l  s  l  l  eh UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ t  t  t  d  ah
REF: iy th ih *  *  sh er t 
HYP: ah sh ih er er er er t 
```

前40段合法状态路径（每帧10ms）：

```text
   0–  10 ms  blank
  10–  20 ms  f:S0
  20–  30 ms  f:S1
  30–  40 ms  f:S2
  40–  50 ms  blank
  50–  70 ms  f:S0
  70– 110 ms  f:S1
 110– 120 ms  f:S2
 120– 130 ms  blank
 130– 140 ms  f:S0
 140– 150 ms  f:S1
 150– 160 ms  f:S2
 160– 170 ms  blank
 170– 180 ms  f:S0
 180– 190 ms  f:S1
 190– 200 ms  f:S2
 200– 210 ms  blank
 210– 220 ms  f:S0
 220– 230 ms  f:S1
 230– 240 ms  f:S2
 240– 250 ms  blank
 250– 260 ms  æ:S0
 260– 270 ms  æ:S2
 270– 300 ms  æ:S0
 300– 310 ms  æ:S2
 310– 320 ms  æ:S0
 320– 430 ms  æ:S1
 430– 440 ms  æ:S2
 440– 450 ms  ɛ:S0
 450– 460 ms  ɛ:S2
 460– 510 ms  t:S0
 510– 540 ms  t:S1
 540– 560 ms  t:S2
 560– 570 ms  t:S0
 570– 580 ms  t:S2
 580– 590 ms  t:S0
 590– 600 ms  t:S2
 600– 620 ms  ɒ:S0
 620– 630 ms  ɒ:S2
 630– 640 ms  ʃ:S0
```

jsd_only / cumulative_average / timit:fadg0_si1279

参考 18 音素，预测 53 音素；S=6 D=0 I=35，PER=227.78%

```text
REF: b r ih k s aa r er n aa l t er n ih t ih v
HYP: ih ay f f f r ih ah ih ih f s s s s s aa aa aa aa r ah n ae n aa ah l l ah t ae er er r eh ah ah ih ih ih ih ih ih ih ih ih n n n n n n
GREEDY: s r r
```

编辑对齐：

```text
REF: *  *  *  *  b  r  ih *  *  *  k  s  *  *  *  *  aa *  *  *  r  er n  *  * 
HYP: ih ay f  f  f  r  ih ah ih ih f  s  s  s  s  s  aa aa aa aa r  ah n  ae n 
REF: aa *  l  *  *  t  *  er *  *  *  *  n  ih t  ih *  *  *  *  *  *  *  *  * 
HYP: aa ah l  l  ah t  ae er er r  eh ah ah ih ih ih ih ih ih ih ih ih n  n  n 
REF: *  *  v 
HYP: n  n  n 
```

前40段合法状态路径（每帧10ms）：

```text
   0–  10 ms  ɪ:S0
  10–  20 ms  ɪ:S2
  20–  30 ms  blank
  30–  50 ms  aj:S0
  50–  60 ms  aj:S2
  60–  70 ms  blank
  70–  80 ms  f:S0
  80–  90 ms  f:S1
  90– 100 ms  f:S2
 100– 110 ms  blank
 110– 130 ms  f:S0
 130– 140 ms  f:S2
 140– 150 ms  f:S0
 150– 160 ms  f:S2
 160– 190 ms  blank
 190– 210 ms  ɹ:S0
 210– 220 ms  ɹ:S2
 220– 230 ms  blank
 230– 250 ms  ɪ:S0
 250– 260 ms  ɪ:S2
 260– 270 ms  ə:S0
 270– 280 ms  ə:S2
 280– 290 ms  ɪ:S0
 290– 300 ms  ɪ:S2
 300– 310 ms  ɪ:S0
 310– 320 ms  ɪ:S1
 320– 330 ms  ɪ:S2
 330– 350 ms  f:S0
 350– 370 ms  f:S1
 370– 380 ms  f:S2
 380– 410 ms  s:S0
 410– 420 ms  s:S2
 420– 430 ms  s:S0
 430– 440 ms  s:S2
 440– 450 ms  blank
 450– 470 ms  s:S0
 470– 480 ms  s:S2
 480– 490 ms  s:S0
 490– 500 ms  s:S2
 500– 510 ms  s:S0
```

jsd_only / cumulative_average / timit:fadg0_si1909

参考 24 音素，预测 87 音素；S=14 D=0 I=63，PER=320.83%

```text
REF: f ae t sh ow d ih n l uw s r ow l z b ih n iy th ih sh er t
HYP: ae ae ay ae ay ae eh eh ae ae ae t ae t ae eh s ae ae ae ae ih ih n n ih ih ih ih ih ih ih ih ih n ih ih ah ih ah ah s s s f s r ah aa ah ah ah s l s l s s ah s t ih ih ih ih s s s f s f ih ah ch ah ih ih ih ih f er r r r er z z
GREEDY: ae ih ih s l l ih r r
```

编辑对齐：

```text
REF: f  ae *  *  *  *  *  *  *  *  *  t  *  *  *  *  *  *  sh ow d  ih *  n  * 
HYP: ae ae ay ae ay ae eh eh ae ae ae t  ae t  ae eh s  ae ae ae ae ih ih n  n 
REF: *  *  *  *  *  *  *  *  *  *  *  *  *  *  l  uw s  *  *  *  *  r  *  *  * 
HYP: ih ih ih ih ih ih ih ih ih n  ih ih ah ih ah ah s  s  s  f  s  r  ah aa ah
REF: *  *  ow l  *  *  *  *  *  z  b  ih *  *  *  *  *  *  n  iy th ih *  *  * 
HYP: ah ah s  l  s  l  s  s  ah s  t  ih ih ih ih s  s  s  f  s  f  ih ah ch ah
REF: *  *  *  *  sh er *  *  *  *  *  t 
HYP: ih ih ih ih f  er r  r  r  er z  z 
```

前40段合法状态路径（每帧10ms）：

```text
   0–  10 ms  æ:S0
  10–  20 ms  æ:S2
  20–  30 ms  blank
  30–  50 ms  æ:S0
  50–  60 ms  æ:S2
  60–  70 ms  blank
  70–  90 ms  aj:S0
  90– 100 ms  aj:S2
 100– 110 ms  blank
 110– 130 ms  æ:S0
 130– 140 ms  æ:S2
 140– 150 ms  blank
 150– 170 ms  aj:S0
 170– 180 ms  aj:S2
 180– 190 ms  blank
 190– 210 ms  æ:S0
 210– 220 ms  æ:S2
 220– 230 ms  blank
 230– 250 ms  ɛ:S0
 250– 260 ms  ɛ:S2
 260– 270 ms  blank
 270– 290 ms  ɛ:S0
 290– 300 ms  ɛ:S2
 300– 310 ms  blank
 310– 320 ms  æ:S0
 320– 340 ms  æ:S2
 340– 350 ms  blank
 350– 360 ms  æ:S0
 360– 380 ms  æ:S2
 380– 390 ms  blank
 390– 400 ms  æ:S0
 400– 420 ms  æ:S2
 420– 430 ms  blank
 430– 450 ms  t:S0
 450– 460 ms  t:S2
 460– 470 ms  æ:S0
 470– 480 ms  æ:S2
 480– 490 ms  t:S0
 490– 500 ms  t:S2
 500– 510 ms  æ:S0
```

ce_fs / final / timit:fadg0_si1279

参考 18 音素，预测 56 音素；S=2 D=0 I=38，PER=222.22%

```text
REF: b r ih k s aa r er n aa l t er n ih t ih v
HYP: b b b b b b r r r ih k k k k k s s s s aa aa r r r r ah n aa l l l t t t er er UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ ih t t t t t t ih v v v v v v v v
GREEDY: b b b b r ih k k k k s s aa r r r ah n aa l t t t er UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ t t t t t v v v v v v v
```

编辑对齐：

```text
REF: b  *  *  *  *  *  r  *  *  ih k  *  *  *  *  s  *  *  *  aa *  r  *  *  * 
HYP: b  b  b  b  b  b  r  r  r  ih k  k  k  k  k  s  s  s  s  aa aa r  r  r  r 
REF: er n  aa l  *  *  t  *  *  er *  *              *              *              n              ih t  *  *  *  *  *  ih *  * 
HYP: ah n  aa l  l  l  t  t  t  er er UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ ih t  t  t  t  t  t  ih v  v 
REF: *  *  *  *  *  v 
HYP: v  v  v  v  v  v 
```

前40段合法状态路径（每帧10ms）：

```text
   0–  50 ms  b:S0
  50–  60 ms  b:S2
  60–  70 ms  b:S0
  70–  80 ms  b:S2
  80–  90 ms  b:S0
  90– 100 ms  b:S2
 100– 110 ms  b:S0
 110– 120 ms  b:S2
 120– 130 ms  b:S0
 130– 140 ms  b:S2
 140– 150 ms  b:S0
 150– 160 ms  b:S2
 160– 170 ms  ɹ:S0
 170– 180 ms  ɹ:S2
 180– 190 ms  ɹ:S0
 190– 200 ms  ɹ:S2
 200– 210 ms  ɹ:S0
 210– 220 ms  ɹ:S2
 220– 270 ms  ɪ:S0
 270– 280 ms  ɪ:S2
 280– 300 ms  blank
 300– 330 ms  k:S0
 330– 340 ms  k:S2
 340– 370 ms  k:S0
 370– 380 ms  k:S2
 380– 390 ms  k:S0
 390– 400 ms  k:S2
 400– 410 ms  k:S0
 410– 420 ms  k:S2
 420– 430 ms  k:S0
 430– 440 ms  k:S2
 440– 450 ms  s:S0
 450– 460 ms  s:S2
 460– 470 ms  s:S0
 470– 480 ms  s:S2
 480– 490 ms  s:S0
 490– 500 ms  s:S2
 500– 510 ms  s:S0
 510– 520 ms  s:S2
 520– 540 ms  blank
```

ce_fs / final / timit:fadg0_si1909

参考 24 音素，预测 74 音素；S=3 D=0 I=50，PER=220.83%

```text
REF: f ae t sh ow d ih n l uw s r ow l z b ih n iy th ih sh er t
HYP: f f f f f f f f ae ae ae k t t t t sh sh sh sh ow ow ow d d d d ih n UNMAPPED_IPA:ʎ UNMAPPED_IPA:ʎ UNMAPPED_IPA:ʎ UNMAPPED_IPA:ʎ UNMAPPED_IPA:ʎ UNMAPPED_IPA:ʎ uw uw s s s s r r r ow ow l l l z b z b b b ih UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ iy th d d ah sh sh er er er er t t
GREEDY: f f f f f ae t t t sh sh ow d d ih UNMAPPED_IPA:ʎ UNMAPPED_IPA:ʎ UNMAPPED_IPA:ʎ UNMAPPED_IPA:ʎ UNMAPPED_IPA:ʎ uw s s r r r l l l b b b b UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ iy th d d ah sh er er t t
```

编辑对齐：

```text
REF: f  *  *  *  *  *  *  *  ae *  *  *  t  *  *  *  sh *  *  *  ow *  *  d  * 
HYP: f  f  f  f  f  f  f  f  ae ae ae k  t  t  t  t  sh sh sh sh ow ow ow d  d 
REF: *  *  ih n  *              *              *              *              *              l              uw *  s  *  *  *  r  *  *  ow *  l  *  *  z 
HYP: d  d  ih n  UNMAPPED_IPA:ʎ UNMAPPED_IPA:ʎ UNMAPPED_IPA:ʎ UNMAPPED_IPA:ʎ UNMAPPED_IPA:ʎ UNMAPPED_IPA:ʎ uw uw s  s  s  s  r  r  r  ow ow l  l  l  z 
REF: b  *  *  *  *  ih *              *              *              *              n              iy th *  *  ih sh *  er *  *  *  *  t 
HYP: b  z  b  b  b  ih UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ iy th d  d  ah sh sh er er er er t  t 
```

前40段合法状态路径（每帧10ms）：

```text
   0–  90 ms  f:S0
  90– 100 ms  f:S2
 100– 110 ms  f:S0
 110– 120 ms  f:S2
 120– 130 ms  f:S0
 130– 140 ms  f:S2
 140– 150 ms  f:S0
 150– 160 ms  f:S2
 160– 170 ms  f:S0
 170– 180 ms  f:S2
 180– 190 ms  f:S0
 190– 200 ms  f:S2
 200– 220 ms  blank
 220– 230 ms  f:S0
 230– 240 ms  f:S2
 240– 250 ms  f:S0
 250– 260 ms  f:S2
 260– 310 ms  æ:S0
 310– 320 ms  æ:S2
 320– 350 ms  æ:S0
 350– 360 ms  æ:S2
 360– 370 ms  blank
 370– 390 ms  æ:S0
 390– 400 ms  æ:S2
 400– 410 ms  k:S0
 410– 420 ms  k:S2
 420– 450 ms  t:S0
 450– 460 ms  t:S2
 460– 470 ms  t:S0
 470– 480 ms  t:S2
 480– 490 ms  t:S0
 490– 500 ms  t:S2
 500– 510 ms  t:S0
 510– 520 ms  t:S2
 520– 540 ms  blank
 540– 550 ms  ʃ:S0
 550– 560 ms  ʃ:S2
 560– 580 ms  blank
 580– 590 ms  ʃ:S0
 590– 600 ms  ʃ:S2
```

ce_fs / cumulative_average / timit:fadg0_si1279

参考 18 音素，预测 32 音素；S=7 D=0 I=14，PER=116.67%

```text
REF: b r ih k s aa r er n aa l t er n ih t ih v
HYP: f f f w r eh ih s s aa aa aa r r eh n aa aa aa l eh er er er UNMAPPED_IPA:ɲ ih UNMAPPED_IPA:ɲ ih ih ih ih th
GREEDY: f f f l UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ ih ih z
```

编辑对齐：

```text
REF: *  *  *  b  r  *  ih k  s  aa *  *  r  *  er n  aa *  *  l  t  er *  *  n             
HYP: f  f  f  w  r  eh ih s  s  aa aa aa r  r  eh n  aa aa aa l  eh er er er UNMAPPED_IPA:ɲ
REF: ih t              ih *  *  *  v 
HYP: ih UNMAPPED_IPA:ɲ ih ih ih ih th
```

前40段合法状态路径（每帧10ms）：

```text
   0–  30 ms  blank
  30–  50 ms  f:S0
  50–  60 ms  f:S2
  60–  70 ms  f:S0
  70–  80 ms  f:S2
  80– 100 ms  blank
 100– 110 ms  f:S0
 110– 120 ms  f:S2
 120– 140 ms  blank
 140– 150 ms  w:S0
 150– 160 ms  w:S2
 160– 180 ms  blank
 180– 190 ms  ɹ:S0
 190– 200 ms  ɹ:S2
 200– 220 ms  blank
 220– 230 ms  ɛ:S0
 230– 240 ms  ɛ:S2
 240– 260 ms  blank
 260– 270 ms  ɪ:S0
 270– 280 ms  ɪ:S2
 280– 460 ms  blank
 460– 470 ms  s:S0
 470– 480 ms  s:S2
 480– 500 ms  blank
 500– 510 ms  s:S0
 510– 520 ms  s:S2
 520– 540 ms  blank
 540– 550 ms  ɑ:S0
 550– 560 ms  ɑ:S2
 560– 580 ms  blank
 580– 590 ms  ɑ:S0
 590– 600 ms  ɑ:S2
 600– 620 ms  blank
 620– 630 ms  ɑ:S0
 630– 640 ms  ɑ:S2
 640– 660 ms  blank
 660– 670 ms  ɹ:S0
 670– 680 ms  ɹ:S2
 680– 700 ms  blank
 700– 710 ms  ɹ:S0
```

ce_fs / cumulative_average / timit:fadg0_si1909

参考 24 音素，预测 52 音素；S=12 D=0 I=28，PER=166.67%

```text
REF: f ae t sh ow d ih n l uw s r ow l z b ih n iy th ih sh er t
HYP: ae ae f f f ae ae ae ae eh sh eh eh ey ey ih ih ih n l uh uw uw z er er aa ow l l z s t ah ah UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ y iy iy iy iy s ah ah eh er er er er er th
GREEDY: ae ae ae sh l l UNMAPPED_IPA:ɲ iy ah
```

编辑对齐：

```text
REF: *  *  f  *  *  ae *  *  *  t  sh *  *  ow d  ih *  *  n  l  *  uw *  *  * 
HYP: ae ae f  f  f  ae ae ae ae eh sh eh eh ey ey ih ih ih n  l  uh uw uw z  er
REF: s  r  ow l  *  z  *  *  *  *  b              ih             n  iy *  *  *  *  th ih sh er *  *  * 
HYP: er aa ow l  l  z  s  t  ah ah UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ y  iy iy iy iy s  ah ah eh er er er er
REF: *  t 
HYP: er th
```

前40段合法状态路径（每帧10ms）：

```text
   0–  10 ms  blank
  10–  30 ms  æ:S0
  30–  40 ms  æ:S2
  40–  60 ms  blank
  60–  70 ms  æ:S0
  70–  80 ms  æ:S2
  80– 100 ms  blank
 100– 110 ms  f:S0
 110– 120 ms  f:S2
 120– 140 ms  blank
 140– 150 ms  f:S0
 150– 160 ms  f:S2
 160– 220 ms  blank
 220– 230 ms  f:S0
 230– 240 ms  f:S2
 240– 260 ms  blank
 260– 270 ms  æ:S0
 270– 280 ms  æ:S2
 280– 300 ms  blank
 300– 310 ms  æ:S0
 310– 320 ms  æ:S2
 320– 340 ms  blank
 340– 350 ms  æ:S0
 350– 360 ms  æ:S2
 360– 380 ms  blank
 380– 390 ms  æ:S0
 390– 400 ms  æ:S2
 400– 420 ms  blank
 420– 430 ms  ɛ:S0
 430– 440 ms  ɛ:S2
 440– 660 ms  blank
 660– 670 ms  ʃ:S0
 670– 680 ms  ʃ:S2
 680– 700 ms  blank
 700– 710 ms  ɛ:S0
 710– 720 ms  ɛ:S2
 720– 740 ms  blank
 740– 750 ms  ɛ:S0
 750– 760 ms  ɛ:S2
 760– 780 ms  blank
```

ce_only / final / timit:fadg0_si1279

参考 18 音素，预测 44 音素；S=3 D=0 I=26，PER=161.11%

```text
REF: b r ih k s aa r er n aa l t er n ih t ih v
HYP: n b w b n ih ih ih ih ih k s s s s s aa er er er ih ih ih ih n aa ah l t ih er UNMAPPED_IPA:ɲ er UNMAPPED_IPA:ɲ ih ih ih ih t ih v v v d
GREEDY: ih s s s
```

编辑对齐：

```text
REF: *  b  *  *  r  ih *  *  *  *  k  s  *  *  *  *  aa r  er *  *  *  *  *  n 
HYP: n  b  w  b  n  ih ih ih ih ih k  s  s  s  s  s  aa er er er ih ih ih ih n 
REF: aa *  l  t  *  er *              *  n              ih *  *  *  t  ih v  *  *  * 
HYP: aa ah l  t  ih er UNMAPPED_IPA:ɲ er UNMAPPED_IPA:ɲ ih ih ih ih t  ih v  v  v  d 
```

前40段合法状态路径（每帧10ms）：

```text
   0–  10 ms  n:S0
  10–  20 ms  n:S2
  20–  30 ms  blank
  30–  40 ms  b:S0
  40–  50 ms  b:S2
  50–  60 ms  w:S0
  60–  70 ms  w:S2
  70–  80 ms  b:S0
  80– 120 ms  b:S2
 120– 130 ms  n:S0
 130– 140 ms  n:S2
 140– 160 ms  ɪ:S0
 160– 170 ms  ɪ:S2
 170– 220 ms  ɪ:S0
 220– 230 ms  ɪ:S2
 230– 250 ms  ɪ:S0
 250– 260 ms  ɪ:S1
 260– 300 ms  ɪ:S2
 300– 310 ms  ɪ:S0
 310– 340 ms  ɪ:S2
 340– 350 ms  ɪ:S0
 350– 370 ms  ɪ:S2
 370– 390 ms  k:S0
 390– 400 ms  k:S2
 400– 420 ms  s:S0
 420– 430 ms  s:S2
 430– 440 ms  s:S0
 440– 450 ms  s:S2
 450– 460 ms  s:S0
 460– 480 ms  s:S2
 480– 490 ms  s:S0
 490– 500 ms  s:S1
 500– 520 ms  s:S2
 520– 530 ms  s:S0
 530– 540 ms  s:S2
 540– 560 ms  ɑ:S0
 560– 570 ms  ɑ:S2
 570– 580 ms  ɚ:S0
 580– 590 ms  ɚ:S2
 590– 600 ms  ɚ:S0
```

ce_only / final / timit:fadg0_si1909

参考 24 音素，预测 77 音素；S=3 D=0 I=53，PER=233.33%

```text
REF: f ae t sh ow d ih n l uw s r ow l z b ih n iy th ih sh er t
HYP: ae f f f f f ae f ae dh ae ae ae ae t t ch sh sh sh sh ow ow ow ow ow d n ih ih ih n n l UNMAPPED_IPA:ʎ l uw uw s s p r ah ah ow ow l ow z z z z ih ah UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ iy iy th d th ah ah ah sh sh UNMAPPED_IPA:c sh UNMAPPED_IPA:c er UNMAPPED_IPA:c er UNMAPPED_IPA:c er t t t
GREEDY: ow ow ow ih
```

编辑对齐：

```text
REF: *  f  *  *  *  *  ae *  *  *  *  *  *  *  t  *  *  sh *  *  *  ow *  *  * 
HYP: ae f  f  f  f  f  ae f  ae dh ae ae ae ae t  t  ch sh sh sh sh ow ow ow ow
REF: *  d  *  ih *  *  n  *  l  *              *  uw *  s  *  *  r  *  *  ow *  l  *  z  * 
HYP: ow d  n  ih ih ih n  n  l  UNMAPPED_IPA:ʎ l  uw uw s  s  p  r  ah ah ow ow l  ow z  z 
REF: *  b  ih *  *              n              iy *  th *  *  *  *  ih sh *  *              *  *              er *              *  *              *  * 
HYP: z  z  ih ah UNMAPPED_IPA:ɲ UNMAPPED_IPA:ɲ iy iy th d  th ah ah ah sh sh UNMAPPED_IPA:c sh UNMAPPED_IPA:c er UNMAPPED_IPA:c er UNMAPPED_IPA:c er t 
REF: *  t 
HYP: t  t 
```

前40段合法状态路径（每帧10ms）：

```text
   0–  20 ms  æ:S0
  20–  30 ms  æ:S2
  30–  40 ms  f:S0
  40–  50 ms  f:S2
  50–  70 ms  f:S0
  70–  80 ms  f:S1
  80–  90 ms  f:S2
  90– 100 ms  f:S0
 100– 120 ms  f:S2
 120– 140 ms  f:S0
 140– 160 ms  f:S2
 160– 170 ms  f:S0
 170– 190 ms  f:S2
 190– 200 ms  æ:S0
 200– 210 ms  æ:S2
 210– 220 ms  f:S0
 220– 230 ms  f:S2
 230– 240 ms  æ:S0
 240– 250 ms  æ:S2
 250– 260 ms  ð:S0
 260– 270 ms  ð:S2
 270– 280 ms  æ:S0
 280– 290 ms  æ:S2
 290– 310 ms  æ:S0
 310– 320 ms  æ:S1
 320– 330 ms  æ:S2
 330– 350 ms  æ:S0
 350– 370 ms  æ:S2
 370– 390 ms  æ:S0
 390– 430 ms  æ:S2
 430– 460 ms  t:S0
 460– 470 ms  t:S2
 470– 490 ms  t:S0
 490– 500 ms  t:S1
 500– 520 ms  t:S2
 520– 530 ms  blank
 530– 540 ms  tʃ:S0
 540– 550 ms  tʃ:S2
 550– 560 ms  ʃ:S0
 560– 570 ms  ʃ:S2
```

ce_only / cumulative_average / timit:fadg0_si1279

参考 18 音素，预测 42 音素；S=10 D=0 I=24，PER=188.89%

```text
REF: b r ih k s aa r er n aa l t er n ih t ih v
HYP: n f f er ih ih ih s s s s s s s l l r r aa ah ih n l ah t t d er er ah ih ih ih ih ih ih ih ih ih d d d
GREEDY: s s s s s ih ih ih
```

编辑对齐：

```text
REF: *  *  b  r  ih *  k  s  *  *  *  *  *  *  *  *  *  *  aa r  er n  aa l  t 
HYP: n  f  f  er ih ih ih s  s  s  s  s  s  s  l  l  r  r  aa ah ih n  l  ah t 
REF: *  *  er *  n  ih t  ih *  *  *  *  *  *  *  *  v 
HYP: t  d  er er ah ih ih ih ih ih ih ih ih ih d  d  d 
```

前40段合法状态路径（每帧10ms）：

```text
   0–  30 ms  blank
  30–  50 ms  n:S0
  50–  60 ms  n:S2
  60–  70 ms  blank
  70–  90 ms  f:S0
  90– 100 ms  f:S2
 100– 110 ms  blank
 110– 130 ms  f:S0
 130– 140 ms  f:S2
 140– 150 ms  blank
 150– 160 ms  ɚ:S0
 160– 170 ms  ɚ:S2
 170– 190 ms  blank
 190– 200 ms  ɪ:S0
 200– 210 ms  ɪ:S2
 210– 220 ms  blank
 220– 230 ms  ɪ:S0
 230– 240 ms  ɪ:S2
 240– 260 ms  blank
 260– 270 ms  ɪ:S0
 270– 280 ms  ɪ:S2
 280– 310 ms  blank
 310– 320 ms  s:S0
 320– 330 ms  s:S2
 330– 340 ms  blank
 340– 360 ms  s:S0
 360– 370 ms  s:S2
 370– 380 ms  blank
 380– 410 ms  s:S0
 410– 420 ms  s:S2
 420– 450 ms  s:S0
 450– 460 ms  s:S2
 460– 470 ms  s:S0
 470– 480 ms  s:S2
 480– 490 ms  s:S0
 490– 500 ms  s:S2
 500– 510 ms  s:S0
 510– 520 ms  s:S2
 520– 550 ms  blank
 550– 560 ms  ɫ:S0
```

ce_only / cumulative_average / timit:fadg0_si1909

参考 24 音素，预测 62 音素；S=12 D=0 I=38，PER=208.33%

```text
REF: f ae t sh ow d ih n l uw s r ow l z b ih n iy th ih sh er t
HYP: ae ae ae ae f ae ae ae ae ae s s sh s t t ih ih ih ih ih ih ah v s s s s l s l s aa ah l s s s s z ah ih ih ih iy ih s s s p s ah ch ch iy ih er er er s z z
GREEDY: sh s s l s s s ih s s er
```

编辑对齐：

```text
REF: *  *  *  *  f  ae *  *  *  *  *  t  sh *  ow d  ih *  *  *  *  *  *  *  * 
HYP: ae ae ae ae f  ae ae ae ae ae s  s  sh s  t  t  ih ih ih ih ih ih ah v  s 
REF: *  *  n  l  *  uw s  r  ow l  *  *  *  *  z  b  ih *  n  iy *  *  *  *  * 
HYP: s  s  s  l  s  l  s  aa ah l  s  s  s  s  z  ah ih ih ih iy ih s  s  s  p 
REF: *  *  *  *  th ih sh er *  *  *  t 
HYP: s  ah ch ch iy ih er er er s  z  z 
```

前40段合法状态路径（每帧10ms）：

```text
   0–  30 ms  blank
  30–  40 ms  æ:S0
  40–  50 ms  æ:S2
  50–  70 ms  blank
  70–  80 ms  æ:S0
  80–  90 ms  æ:S2
  90– 110 ms  blank
 110– 120 ms  æ:S0
 120– 130 ms  æ:S2
 130– 150 ms  blank
 150– 160 ms  æ:S0
 160– 170 ms  æ:S2
 170– 190 ms  blank
 190– 210 ms  f:S0
 210– 220 ms  f:S2
 220– 230 ms  blank
 230– 240 ms  æ:S0
 240– 250 ms  æ:S2
 250– 270 ms  blank
 270– 280 ms  æ:S0
 280– 290 ms  æ:S2
 290– 310 ms  blank
 310– 320 ms  æ:S0
 320– 330 ms  æ:S2
 330– 350 ms  blank
 350– 360 ms  æ:S0
 360– 370 ms  æ:S2
 370– 460 ms  blank
 460– 470 ms  æ:S0
 470– 480 ms  æ:S2
 480– 500 ms  blank
 500– 510 ms  s:S0
 510– 520 ms  s:S2
 520– 540 ms  blank
 540– 550 ms  s:S0
 550– 560 ms  s:S2
 560– 590 ms  blank
 590– 610 ms  ʃ:S0
 610– 620 ms  ʃ:S2
 620– 630 ms  s:S0
```
