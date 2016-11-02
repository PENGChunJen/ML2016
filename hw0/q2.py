import sys
from PIL import Image
import numpy as np
from PIL import ImageChops

def main(argv):
    inputFile = argv[1]
    outputFile = 'ans2.png'
    try:
        im = Image.open(inputFile)
        out = im.transpose(Image.ROTATE_180)
        #out = im.rotate(180)
        out.save(outputFile)
        #test_equal(im, out.rotate(180))
    except IOError:
        print 'Cannot open', inputFile 

def test_equal(im1, im2):
    print np.asarray(im1), im1.size
    print np.asarray(im2), im2.size
    print 'Equal?', (ImageChops.difference(im1, im2).getbbox() is None)

if __name__ == "__main__":
    main(sys.argv)
