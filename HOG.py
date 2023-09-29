import cv2
import imutils
import numpy as np
import argparse

def detect(frame):
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)

    person = 1
    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'person {person}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        person += 1

    cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons : {person-1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)

    return frame

def detectByPathVideo(path, output_path):
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if not check:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    print('Detecting people...')
    while True:
        check, frame = video.read()

        if not check:
            break

        frame = imutils.resize(frame, width=min(800, frame.shape[1]))
        frame = detect(frame)

        if output_path is not None:
            out.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    if output_path is not None:
        out.release()

def detectByCamera():
    video = cv2.VideoCapture(0)
    print('Detecting people...')

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

    while True:
        check, frame = video.read()

        frame = detect(frame)

        out.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    out.release()

def detectByPathImage(path, output_path):
    image = cv2.imread(path)

    image = imutils.resize(image, width=min(800, image.shape[1]))

    result_image = detect(image)

    if output_path is not None:
        cv2.imwrite(output_path, result_image)

def humanDetector(args):
    image_path = args["image"]
    video_path = args["video"]
    camera = args["camera"]

    if camera:
        print('[INFO] Opening Web Cam.')
        detectByCamera()
    elif video_path is not None:
        print('[INFO] Opening Video from path.')
        detectByPathVideo(video_path, args["output"])
    elif image_path is not None:
        print('[INFO] Opening Image from path.')
        detectByPathImage(image_path, args["output"])

def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
    arg_parse.add_argument("-i", "--image", default=None, help="path to Image File ")
    arg_parse.add_argument("-c", "--camera", default=False, help="Set true if you want to use the camera.")
    arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")
    args = vars(arg_parse.parse_args())

    return args

if __name__ == "__main__":
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    args = {
        "video": None,  # Video dosyasının yolu
        "image": None,  # Resim dosyasının yolu
        "camera": False,  # Kamera kullanılıp kullanılmayacağı
        "output": None  # Çıkış video dosyasının yolu (isteğe bağlı)
    }

    args = argsParser()
    humanDetector(args)

