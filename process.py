import cv2
import json
def Draw(image, org, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    image = cv2.putText(image, text, org, font,
                      fontScale, color, thickness, cv2.LINE_AA)
    return image


def VideoFrameExtract(vid):
    vid_capture = cv2.VideoCapture(vid)
    frame_width = int(vid_capture.get(3))
    frame_height = int(vid_capture.get(4))
    frame_size = (frame_width, frame_height)
    fps = int(vid_capture.get(5))
    count = 0

    f = open('data/prediction.json')
    data = json.load(f)

    # Initialize video writer object
    out = cv2.VideoWriter('output/distance/hkvideo3.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

    while (vid_capture.isOpened()):
        # vid_capture.read() methods returns a tuple, first element is a bool
        # and the second is frame

        ret, frame = vid_capture.read()
        if ret == True:
            if (data['frame'+str(count)] != {} ):
                for i in range(len(data['frame' + str(count)])):
                    x = int(data['frame' + str(count)]['box' + str(i)]['xyxy'][2])
                    y = int(data['frame' + str(count)]['box' + str(i)]['xyxy'][3])

                    dis_org = (x, y)
                    d = str(int(data['frame' + str(count)]['box' + str(i)]['distance(m)'])) + 'm'
                    frame = Draw(frame, dis_org, d)

                    dir_org = (x, y-50)
                    angle = str(int(data['frame' + str(count)]['box' + str(i)]['direction(degrees)']))
                    frame = Draw(frame, dir_org, angle)

            # Write the frame to the output files
            count += 1
            out.write(frame)

        else:
            break

    vid_capture.release()
    out.release()

def main():
    video_path = 'output/object/hkvideo3.mp4'
    VideoFrameExtract(video_path)

if __name__ == "__main__":
    main()