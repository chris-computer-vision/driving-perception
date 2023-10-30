import math as m

def DirectionPrediction(center_x, center_y, image_size):
    mid = image_size[1] / 2

    if ((center_y - mid) > 0):
        angle = m.atan((image_size[0] - center_x) / (center_y - mid))
        angle = m.degrees(angle)
    else:
        angle = m.atan((image_size[0] - center_x) / (mid - center_y))
        angle = m.degrees(angle) + 90
    direction = angle
    return direction
