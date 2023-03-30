import cv2
import os

def video2images(source_path, dest_dir_path, max_frame):  
    cam = cv2.VideoCapture(source_path)

    try:
        if not os.path.exists(dest_dir_path):
            os.makedirs(dest_dir_path)
    except OSError:
        print ('Error: Creating directory of data')

    currentframe = 0

    while(True):
        ret, frame = cam.read()
        if ret and currentframe < max_frame:
            # if video is still left continue creating images
            name = dest_dir_path + '/frame_{:05d}'.format(currentframe) + '.jpg'
            print ('Creating...' + name)

            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break

    cam.release()
    cv2.destroyAllWindows()

video2images('../data/sports.mp4', '../data/frames_sports', 20000)
