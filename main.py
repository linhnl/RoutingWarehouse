import numpy as np
from routing_map import Routing

from constants import MAP_PATH
import animation 
import cv2

def main():
    routing = Routing(MAP_PATH) 
    # routing.Map[15, 6] = 2
    # print(routing.Map[15, 6])

    animation.animation(routing)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()