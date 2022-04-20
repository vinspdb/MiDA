from read_log import ReadLog
from MiDA import MiDA
import sys
if __name__ == "__main__":
    get_log = ReadLog(sys.argv[1]).readView()
    MiDA(sys.argv[1]).smac_opt()

