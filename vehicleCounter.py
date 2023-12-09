#### IMPORTACOES, ABERTURA DE ARQUIVOS E SETAGEM DE VARIÁVEIS

import cv2 # biblioteca de manipulação de imagens
import torch
import numpy as np
from ultralytics import YOLO # modelo para detecção de objetos
#from tracker import *

def VehicleCounting (caminho):
  video = cv2.VideoCapture(caminho) # abertura do video

  ret, frame = video.read() #separação do vídeo em frames, se deu tudo certo, ret == True
  
  count = 0


  #### CARREGAMENTO DE MODELO E TRACKER

  model = YOLO("yolov8n.pt") # modelo pronto da biblioteca ultralytics para detecção de diversos objetos, versão 8, possível de iterar sobre resultado
  tracker = Tracker()

  #### DETECÇÃO


  while ret: # como temos uma coleção de frames, devemos iterar sobre todos eles.
    results = model(frames) # rodo o modelo sobre um frame e armazeno em results
    result = results[0] # na primeira linha está armazenado
    detection_threshold = 0.5
    detections=[]

    for r in result.boxes.data.tolist(): #itero sobre todos os objetos detectados, salvando de cada um

              x1, y1, x2, y2, score, class_id = r # a posição de ínicio e fim do objeto em x e y, o score e qual objeto é
                                                  # o score retornado indica a confiança
              x1 = int(x1) # convertendo para inteiro
              x2 = int(x2)
              y1 = int(y1)
              y2 = int(y2)
              class_id = int(class_id) 
              if score > detection_threshold: # descarta-se objetos com confiança menor que 0.5
                  detections.append([x1, y1, x2, y2, score]) # adiciono o resultado no dataframe de objetos detectados.
    tracker.update(frames, detections) # adiciono o objeto no tracker para acompanhá-lo ao longo do vídeo

    # função para visalizar boxes nos objetos encontrados e contar quantos foram encontrados
    for track in tracker.tracks:
      bbox = track.bbox
      track_id = track.track_id
      cv2.retangle(frames, (x1,y1), (x2,y2), colors = 255)
      count+=1 # qtd de veículos encontrados
    return count