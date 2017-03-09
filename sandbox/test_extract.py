import extract

def test_extract():
    tpf = extract.TargetPixelFile('ktwo210459199-c04_lpd-targ.fits.gz')
    mask = tpf.pixel_mask()
