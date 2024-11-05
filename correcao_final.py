import face_recognition
import cv2
import os
import time

# Diretório onde estão as fotos

# descomentar ao testar local
#diretorio_fotos = 'fotos/'
#diretorio_model = 'face_recognition_models/models/shape_predictor_68_face_landmarks.dat'

# descomentar ao gerar executável
diretorio_fotos = '_internal\\fotos'
diretorio_model = '_internal\\face_recognition_models\\models\\shape_predictor_68_face_landmarks.dat'

# Carregar o arquivo de modelo
os.environ['FACE_RECOGNITION_MODEL_LOCATION'] = diretorio_model

# Verificar se o arquivo do modelo existe
if not os.path.isfile(diretorio_model):
    raise RuntimeError(f"Arquivo do modelo não encontrado em: {diretorio_model}")

# Lista para armazenar as codificações faciais dos alunos
alunos_face_encodings = []
# Lista para armazenar os nomes dos alunos
alunos_names = []

# Carregar as imagens e codificações faciais de todos os alunos
for filename in os.listdir(diretorio_fotos):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        aluno_image = face_recognition.load_image_file(os.path.join(diretorio_fotos, filename))
        aluno_face_encoding = face_recognition.face_encodings(aluno_image)[0]
        alunos_face_encodings.append(aluno_face_encoding)
        alunos_names.append(os.path.splitext(filename)[0])

# Inicializar a câmera
cap = cv2.VideoCapture(0)

# Definir o tamanho da janela de exibição
window_width = 1200
window_height = 800
cv2.namedWindow('Reconhecimento Facial', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Reconhecimento Facial', window_width, window_height)

# Variável para controlar o estado de presença dos alunos
alunos_presentes = {name: False for name in alunos_names}
# Variável para controlar o estado da mensagem de saída
mensagem_saiu_exibida = {name: False for name in alunos_names}

# Variável para controlar o tempo da última verificação
ultima_verificacao = time.time()

# Definir o tempo de verificação em segundos
tempo_def = 20

# Criar o nome do arquivo com base na data atual
data_atual = time.strftime('%d-%m-%Y')
nome_arquivo = f'presença-{data_atual}.txt'

# Abrir o arquivo para escrita
with open(nome_arquivo, 'a') as arquivo:
    print(f"\nLocal do modelo: {os.environ['FACE_RECOGNITION_MODEL_LOCATION']}")
    arquivo.write(f"\nLocal do modelo: {os.environ['FACE_RECOGNITION_MODEL_LOCATION']}")

    print(f"\nTempo definido (em segundos): {tempo_def}\n\n")
    arquivo.write(f"\nTempo definido (em segundos): {tempo_def}\n\n")

    estado_atual = f"\n[+] Estado atual de presença dos alunos ( {time.strftime('%H:%M:%S')} ):\n"
    print(estado_atual)
    arquivo.write(estado_atual)
    for name in alunos_names:
        status = "Ausente"
        estado_aluno = f"{name}: {status}"
        print(estado_aluno)
        arquivo.write(estado_aluno + '\n')
    print("\n")
    arquivo.write("\n")

    while True:
        # Capturar um quadro da câmera
        ret, frame = cap.read()

        # Encontrar todas as faces no quadro
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Obter o tempo atual do sistema
        tempo_atual = time.strftime('%Y-%m-%d %H:%M:%S')

        # Variável para controlar quais alunos estão na sala no momento
        alunos_atualmente_presentes = set()

        # Loop pelas faces encontradas
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(alunos_face_encodings, face_encoding)
            name = "Desconhecido"

            # Verificar se encontramos um match
            face_distances = face_recognition.face_distance(alunos_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = alunos_names[best_match_index]
                alunos_atualmente_presentes.add(name)
                if not alunos_presentes[name]:
                    msg = f"[+] {name} entrou na sala : {tempo_atual}"
                    print(msg)
                    arquivo.write(msg + '\n')
                    alunos_presentes[name] = True
                    mensagem_saiu_exibida[name] = False
                    # Exibir mensagem no canto direito em cor verde
                    cv2.putText(frame, f"{name} entrou na sala", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Desenhar um retângulo ao redor da face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Exibir o nome
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Verificar quais alunos saíram da sala
        for name in alunos_names:
            if alunos_presentes[name] and name not in alunos_atualmente_presentes:
                msg = f"[-] {name} saiu da sala : {tempo_atual}"
                print(msg)
                arquivo.write(msg + '\n')
                mensagem_saiu_exibida[name] = True
                alunos_presentes[name] = False
                # Exibir mensagem na própria exibição em cor vermelha
                cv2.putText(frame, f"{name} saiu da sala", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        def verificar():
            # Imprimir o estado atual de presença dos alunos
            estado_atual = f"\n[+] Estado atual de presença dos alunos ( {time.strftime('%H:%M:%S')} ):\n"
            print(estado_atual)
            arquivo.write(estado_atual)
            for name in alunos_names:
                status = "Presente" if alunos_presentes[name] else "Ausente"
                estado_aluno = f"{name}: {status}"
                print(estado_aluno)
                arquivo.write(estado_aluno + '\n')
            print("\n")
            arquivo.write("\n")

        # Verificar se o tempo_def se passou desde a última impressão
        if time.time() - ultima_verificacao >= tempo_def:
            verificar()

            # Atualizar o tempo da última verificação
            ultima_verificacao = time.time()

        # Exibir o quadro resultante
        cv2.imshow('Reconhecimento Facial', frame)

        # Encerrar o aplicativo quando pressionar a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            verificar()
            break

# Liberar os recursos e fechar a janela
cap.release()
cv2.destroyAllWindows()
