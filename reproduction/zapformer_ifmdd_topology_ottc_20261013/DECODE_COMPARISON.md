# Subphonetic topology OTTC: fixed training utterance decode versus MFA reference

200 updates，四条真实训练语音，共 60 个 reference tokens。greedy 与 S3T2** FSA decode 均 0% PER。仅用于验证可学习性，不是 held-out benchmark。

## librispeech:4014-186175-0030-28192

Reference: `ɲ ɪ ɹ ʎ i s ɐ f ə cʰ ej ɾʲ ɪ ɾ tʰ ʉ`

IF-MDD greedy: `ɲ ɪ ɹ ʎ i s ɐ f ə cʰ ej ɾʲ ɪ ɾ tʰ ʉ`

S3T2** FSA: `ɲ ɪ ɹ ʎ i s ɐ f ə cʰ ej ɾʲ ɪ ɾ tʰ ʉ`

S/D/I = 0/0/0。

## librispeech:3242-67168-0039-6071

Reference: `aj s ej ɫ f ɚ h ow m t ə m ɒ ɹ ow`

IF-MDD greedy: `aj s ej ɫ f ɚ h ow m t ə m ɒ ɹ ow`

S3T2** FSA: `aj s ej ɫ f ɚ h ow m t ə m ɒ ɹ ow`

S/D/I = 0/0/0。

## librispeech:403-126855-0000-2193

Reference: `d̪ ə s ɛ v ə n θ tʃ æ p t ɚ`

IF-MDD greedy: `d̪ ə s ɛ v ə n θ tʃ æ p t ɚ`

S3T2** FSA: `d̪ ə s ɛ v ə n θ tʃ æ p t ɚ`

S/D/I = 0/0/0。

## librispeech:3259-158083-0005-27165

Reference: `mʲ ɪ s ɪ z ɛ ɫ ej tʃ h ɒ ɹ n z bʲ i`

IF-MDD greedy: `mʲ ɪ s ɪ z ɛ ɫ ej tʃ h ɒ ɹ n z bʲ i`

S3T2** FSA: `mʲ ɪ s ɪ z ɛ ɫ ej tʃ h ɒ ɹ n z bʲ i`

S/D/I = 0/0/0。
