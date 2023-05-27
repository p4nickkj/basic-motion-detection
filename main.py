
import cv2
import time

fps = 50
tempo_gravacao = 5
largura, altura = 640, 480

def reiniciar_captura():
    global captura
    captura.release()
    time.sleep(2)
    captura = cv2.VideoCapture(0)
    captura.set(3, largura)
    captura.set(4, altura)

class MotionDetector:
    def __init__(self, largura, altura, tempo_gravacao, fps):
        self.largura = largura
        self.altura = altura
        self.tempo_gravacao = tempo_gravacao
        self.fps = fps
        self.subtrator = cv2.createBackgroundSubtractorMOG2()

    def processar_quadro(self, quadro):
        mascarabg = self.subtrator.apply(quadro)
        mascarabg = cv2.threshold(mascarabg, 254, 255, cv2.THRESH_BINARY)[1]
        mascarabg = cv2.erode(mascarabg, None, iterations=2)
        mascarabg = cv2.dilate(mascarabg, None, iterations=2)

        contornos, _ = cv2.findContours(mascarabg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        movimento_detectado = False

        for contorno in contornos:
            if cv2.contourArea(contorno) > 500:
                (x, y, w, h) = cv2.boundingRect(contorno)
                cv2.rectangle(quadro, (x, y), (x + w, y + h), (255, 0, 255), 2)
                movimento_detectado = True

        return movimento_detectado

detector = MotionDetector(largura, altura, tempo_gravacao, fps)
gravando = False
tempo_inicio = None
video_writer = None

captura = cv2.VideoCapture(0)
captura.set(3, largura)
captura.set(4, altura)

while True:
    try:
        ret, quadro = captura.read()

        if not ret:
            print("Erro na leitura do quadro. Reiniciando...")
            reiniciar_captura()
            continue

        movimento_detectado = detector.processar_quadro(quadro)

        if movimento_detectado:
            if not gravando:
                tempo_inicio = time.time()
                gravando = True

        elif gravando:
            if time.time() - tempo_inicio >= tempo_gravacao:
                gravando = False

        if gravando:
            if video_writer is None:
                nome_arquivo = f"gravacao_{int(time.time())}.mp4"
                video_writer = cv2.VideoWriter(nome_arquivo, cv2.VideoWriter_fourcc(*'mp4v'), fps, (largura, altura))
            video_writer.write(quadro)

        cv2.imshow("Detecção de Movimento", quadro)

        # Verificar se a tecla 'ESC' foi pressionada
        if cv2.waitKey(1) == 27:
            break

    except cv2.error as exception:
        print("Ocorreu um erro:", exception)
        print("A reiniciar o programa...")
        reiniciar_captura()
        gravando = False
        tempo_inicio = None
        if video_writer is not None:
            video_writer.release()
        video_writer = None
        continue

captura.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()
