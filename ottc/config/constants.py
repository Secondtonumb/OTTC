# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kaloga Yacouba yacouba.kaloga@idiap.ch
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch
#
# SPDX-License-Identifier: MIT

"""
Project-wide scalar constants.
"""
# this token used for representing space (default behavior of Wav2Vec2Tokenizer)
PIPE_TOKEN = ("|", 4)

# Token used when splitting double letter (OTTC)
SPLITTING_DOUBLE_LETTER_TOKEN = ("*", 3)


# token used ft for bordering tokens (OTTC)
BORDERING_TOKEN = ("#", 5)
