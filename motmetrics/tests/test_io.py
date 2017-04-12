import motmetrics as mm
import pandas as pd

def test_load_vatic():
    df = mm.io.loadtxt('etc/data/iotest/vatic.txt', fmt=mm.io.Format.VATIC_TXT)

    expected = pd.DataFrame([
        #F,ID,Y,W,H,L,O,G,F,A1,A2,A3,A4
        (0,0,412,0,430,124,0,0,0,'worker',0,0,0,0),
        (1,0,412,10,430,114,0,0,1,'pc',1,0,1,0),
        (1,1,412,0,430,124,0,0,1,'pc',0,1,0,0),
        (2,2,412,0,430,124,0,0,1,'worker',1,1,0,1)
    ])

    assert (df.reset_index().values == expected.values).all()