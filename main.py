import cv2
import time

fps = 60
tempo_gravacao = 5
largura, altura = 640, 480

captura = cv2.VideoCapture(0)
captura.set(3, largura)
captura.set(4, altura)

class MotionDetector:
    def __init__(self, largura, altura, tempo_gravacao, fps):
        self.largura = largura
        self.altura = altura
        self.tempo_gravacao = tempo_gravacao
        self.fps = fps
        self.quadro_anterior = None
        self.tempo_inicio = None
        self.gravando = False
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

        if movimento_detectado:
            if not self.gravando:
                self.tempo_inicio = time.time()
                self.gravando = True

        elif self.gravando:
            if time.time() - self.tempo_inicio >= self.tempo_gravacao:
                self.gravando = False

    def reset(self):
        self.quadro_anterior = None
        self.tempo_inicio = None
        self.gravando = False

detector = MotionDetector(largura, altura, tempo_gravacao, fps)

while True:
    try:
        ret, quadro = captura.read()

        detector.processar_quadro(quadro)

        if detector.gravando:
            if 'video_writer' not in locals():
                nome_arquivo = f"video_{int(time.time())}.mp4"
                video_writer = cv2.VideoWriter(nome_arquivo, cv2.VideoWriter_fourcc(*'mp4v'), fps, (largura, altura))
            video_writer.write(quadro)

        cv2.imshow("Detecção de Movimento", quadro)

        # Verificar se a tecla 'ESC' foi pressionada
        if cv2.waitKey(1) == 27:
            break

    except cv2.error as exception:
        print("Ocorreu um erro:", exception)
        print("A reiniciar o programa...")
        captura.release()
        cv2.destroyAllWindows()
        time.sleep(2)
        captura = cv2.VideoCapture(0)
        captura.set(3, largura)
        captura.set(4, altura)
        detector.reset()
        continue

captura.release()
if 'video_writer' in locals():
    video_writer.release()
cv2.destroyAllWindows()
