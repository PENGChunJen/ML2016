import sys
import numpy as np

def main(argv):
    col = int(argv[1])
    inputFile = argv[2]
    outputFile = 'ans1.txt'
    try:
        f = open(inputFile, 'r')
        arr = np.loadtxt(f)
        ans = sorted(arr.T[col])
        np.savetxt(outputFile, ans, newline=',') 
    except IOError:
        print 'cannot open', inputFile 
    except IndexError:
        print col, 'is out of range'
        
if __name__ == "__main__":
    main(sys.argv)
