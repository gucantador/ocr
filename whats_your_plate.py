import cv2
import os
import numpy as np
import pytesseract
import easyocr



pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

print(pytesseract.get_tesseract_version())

PATH = "placas_exercicio/"
PATH_CUT = 'placas_recortadas/'
PATH_DEBUG = 'debug_images/'

if not os.path.isdir(PATH_DEBUG):
    os.mkdir(PATH_DEBUG)

def detect_plates(path:str) -> list:
    imagem = cv2.imread(path)

    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    )
    placas = detector.detectMultiScale(cinza, 1.1, 5, minSize=(60, 20))
    for i, (x, y, w, h) in enumerate(placas):
        roi = imagem[y:y+h, x:x+w]
        if not os.path.isdir("placas_recortadas"):
            os.mkdir("placas_recortadas")
        nome_imagem = os.path.splitext(os.path.basename(path))[0]

        cv2.imwrite(
            f"placas_recortadas/{nome_imagem}_placa.png",
            roi
        )
        cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return placas




def pre_proc(path, binary_type='simples', blur=True, morph=True):
    # cinza
    imagem = cv2.imread(path)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


    if binary_type == "simples":
        _, imagem = cv2.threshold(
            imagem,
            127,
            255,
            cv2.THRESH_BINARY
            )
    if binary_type == 'otsu':
    # binarizar threshold otsu
        _, imagem = cv2.threshold(
                imagem,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
    if binary_type == "adaptativo":
        imagem = cv2.adaptiveThreshold(
                imagem,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11
                )

    
    if blur:
        imagem = cv2.GaussianBlur(
                imagem,
                (5, 5),
                0
                )


    if morph:
        kernel = np.ones((3, 3), np.uint8)
        imagem = cv2.morphologyEx(
        imagem,
        cv2.MORPH_CLOSE,
        kernel
        )

    nome_arquivo = os.path.basename(path)

    cv2.imwrite(
        os.path.join(
            PATH_DEBUG,
            f"{binary_type}_blur{blur}_morph{morph}_{nome_arquivo}"
        ),
        imagem
    )

    return imagem


def ocr_tesseract(imagem):

    texto = pytesseract.image_to_string(
    imagem,
    config="--psm 7"
    )
    print(repr(texto))




# 1. Detecta placas em todas as imagens da pasta
for arquivo in os.listdir(PATH):
    if arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
        caminho = os.path.join(PATH, arquivo)
        detect_plates(caminho)

# 2. OCR em todos os recortes gerados
resultados = []



binary_types = ["simples", "otsu", "adaptativo"]

for arquivo in os.listdir(PATH_CUT):
    if arquivo.lower().endswith(".png"):

        caminho = os.path.join(PATH_CUT, arquivo)

        resultados.append("=" * 80)
        resultados.append(f"ARQUIVO: {arquivo}")
        resultados.append("=" * 80)

        for binary_type in binary_types:
            for blur in [False, True]:
                for morph in [False, True]:

                    try:
                        img = pre_proc(
                            caminho,
                            binary_type=binary_type,
                            blur=blur,
                            morph=morph
                        )

                        texto = pytesseract.image_to_string(
                            img,
                            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                        ).strip()

                        resultados.append(
                            f"binary={binary_type:<10} "
                            f"blur={str(blur):<5} "
                            f"morph={str(morph):<5} "
                            f"-> '{texto}'"
                        )

                    except Exception as e:
                        resultados.append(
                            f"binary={binary_type} "
                            f"blur={blur} "
                            f"morph={morph} "
                            f"-> ERRO: {e}"
                        )

        resultados.append("")

with open("resultado_tesseract_ocr.txt", "w", encoding="utf-8") as f:
    for linha in resultados:
        f.write(linha + "\n")



print("tesseract_concluido")

reader = easyocr.Reader(["en"], gpu=False)

resultados = []

binary_types = ["simples", "otsu", "adaptativo"]

for arquivo in os.listdir(PATH_CUT):
    if arquivo.lower().endswith(".png"):

        caminho = os.path.join(PATH_CUT, arquivo)

        resultados.append("=" * 80)
        resultados.append(f"ARQUIVO: {arquivo}")
        resultados.append("=" * 80)

        for binary_type in binary_types:
            for blur in [False, True]:
                for morph in [False, True]:

                    try:

                        img = pre_proc(
                            caminho,
                            binary_type=binary_type,
                            blur=blur,
                            morph=morph
                        )

                        print(
                            f"Processando {arquivo} | "
                            f"binary={binary_type} "
                            f"blur={blur} "
                            f"morph={morph}"
                        )

                        resultado = reader.readtext(
                            img,
                            detail=1,
                            paragraph=False
                        )

                        if resultado:
                            texto = " | ".join(
                                [item[1] for item in resultado]
                            )
                        else:
                            texto = ""

                        resultados.append(
                            f"binary={binary_type:<10} "
                            f"blur={str(blur):<5} "
                            f"morph={str(morph):<5} "
                            f"-> '{texto}'"
                        )

                    except Exception as e:

                        resultados.append(
                            f"binary={binary_type:<10} "
                            f"blur={str(blur):<5} "
                            f"morph={str(morph):<5} "
                            f"-> ERRO: {e}"
                        )

        resultados.append("")

with open("resultado_easyocr.txt", "w", encoding="utf-8") as f:
    for linha in resultados:
        f.write(linha + "\n")

print("EasyOCR concluído.")