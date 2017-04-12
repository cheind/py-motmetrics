import motmetrics as mm

def test_load_vatic():
    print(mm.io.loadtxt('etc/data/vatic.txt', fmt=mm.io.Format.VATIC_TXT))