import dlib
import cv2

detector = dlib.get_frontal_face_detector()

image_path = 'test_data/test.jpg'
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = detector(gray)

for face in faces:
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
