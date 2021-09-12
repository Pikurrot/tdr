import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = \
    "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"

imatge = cv2.imread("data/OCR_text2.png")
alçada,amplada,_ = imatge.shape
r = min(800/alçada,1)
imatge = cv2.resize(imatge, (round(amplada * r), round(alçada * r)))
cv2.imshow("Original", imatge)
escala_de_grisos = cv2.cvtColor(imatge, cv2.COLOR_BGR2GRAY)
cv2.imshow("Escala de grisos", escala_de_grisos)
gaussian = cv2.adaptiveThreshold(escala_de_grisos,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,111,20)
cv2.imshow("Gaussian", gaussian)

marcs = pytesseract.image_to_data(gaussian)
for i,b in enumerate(marcs.splitlines()):
    if i!=0:
        b = b.split()
        if len(b) == 12:
            x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
            cv2.rectangle(imatge, (x,y), (w+x,h+y), (0,0,255), 1)
            cv2.putText(imatge, b[11], (x, y - 5), 
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0 ,255), 1)

print(pytesseract.image_to_string(gaussian))
cv2.imshow("Prediccio", imatge)
cv2.waitKey()


