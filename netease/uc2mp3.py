# -*- coding: utf-8 -*-
with open ('0.uc','rb') as f:
    btay = bytearray(f.read())
with open('test.mp3','wb') as out:
    for i,j in enumerate(btay):
        btay[i] = j ^ 0xa3
        print (btay[i])
        print (j)
        break
    out.write(bytes(btay))
