import cv2


def main():
    # Încărcarea clasificatorului Haar Cascade pentru detectarea fețelor
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Citirea imaginii de test
    img = cv2.imread('test_data/test3.jpeg')
    if img is None:
        print("Nu s-a putut încărca imaginea.")
        return

    # Conversia imaginii în nuanțe de gri
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectarea fețelor în imagine
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenarea dreptunghiurilor în jurul fețelor detectate
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Afișarea imaginii cu fețele detectate
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
