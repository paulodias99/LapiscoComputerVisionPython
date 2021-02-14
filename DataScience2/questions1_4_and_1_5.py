###
#
# Pelos códigos 4 e 5 reutilizarem os códigos antigos, normalizarem as imagens e gerar uma moda
# utilizando as imagens normalizadas preferi uní-los num só .py
#
###

import glob
import os

#importando codigos antigos para usar as funções dos outros códigos
from question1_2 import RecoveryImage
#from question1_3 import RecoveryImage

#importando as funções de normalização e da moda gerada
from black_box_normalization import *
from mode_chain_codes import *

#black_box = BlackBox(chain_codes, signalLenght)

#mode_chain_codes = ModeChainCode(black_box.new_chain_codes)

draw_recovery = RecoveryImage(mode_chain_codes.chain_code_mode)