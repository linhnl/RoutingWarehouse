import numpy as np
from routing_map import Routing

from constants import MAP_PATH
import animation 
import cv2

def main():
    # warehouse = routing_map.load_map(MAP_PATH)
    # print(warehouse.shape)
    # drawn_map = animation.draw_map(warehouse)
    # cv2.imshow('drawn_map', drawn_map)
    # cv2.waitKey(0)
    routing = Routing(MAP_PATH)
    # routing.__init__(MAP_PATH)
    animation.animation(routing)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()