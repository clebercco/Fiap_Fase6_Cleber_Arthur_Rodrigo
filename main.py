# Passo 1: Importar bibliotecas e montar o Google Drive
import os
import shutil
from google.colab import drive
import yaml
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Tentar montar o Google Drive
try:
    drive.mount('/content/drive', force_remount=True)
    print("Google Drive montado com sucesso!")
except Exception as e:
    print(f"Erro ao montar Google Drive: {e}")
    raise SystemExit("Por favor, verifique a conexão com o Google Drive e tente novamente.")

# Resetar o diretório de trabalho
os.chdir('/content')
print(f"Diretório de trabalho atual: {os.getcwd()}")

# Função para verificar diretórios
def check_directory(path, name):
    if os.path.exists(path):
        print(f"{name} encontrado: {path}")
        print(f"Conteúdo de {name}:")
        !ls -l {path}
        return True
    else:
        print(f"ERRO: {name} não encontrado em {path}")
        return False

# Definir diretórios
base_dir = "/content/drive/MyDrive/Fiap/Fase6"
train_image_dir = f"{base_dir}/train/Imagens"
train_label_dir = f"{base_dir}/train/labels"
val_image_dir = f"{base_dir}/val/Imagens"
val_label_dir = f"{base_dir}/val/labels"
test_image_dir = f"{base_dir}/test/Imagens"
subset_train_image_dir = "/content/test_dataset/train/Imagens"
subset_train_label_dir = "/content/test_dataset/train/labels"

# Verificar diretórios
print("\nVerificando diretórios...")
base_dir_exists = check_directory(base_dir, "Diretório base (Fiap/Fase6)")
if not base_dir_exists:
    print("Possíveis caminhos alternativos:")
    !ls -R /content/drive/MyDrive
    print("\nVerificando Shared drives:")
    !ls -R /content/drive/Shared\ drives
    raise SystemExit("Por favor, verifique o caminho do diretório 'Fiap/Fase6' no Google Drive e atualize o script.")

check_directory(train_image_dir, "Diretório de imagens de treinamento")
check_directory(train_label_dir, "Diretório de rótulos de treinamento")
check_directory(val_image_dir, "Diretório de imagens de validação")
check_directory(val_label_dir, "Diretório de rótulos de validação")
check_directory(test_image_dir, "Diretório de imagens de teste")

# Função para criar arquivos de rótulo vazios (para YOLOv5)
def create_missing_labels(image_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    image_extensions = (".jpg", ".jpeg", ".png")
    missing_labels = []
    for image_file in os.listdir(image_dir):
        if image_file.endswith(image_extensions):
            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(label_dir, label_file)
            if not os.path.exists(label_path):
                print(f"Criando arquivo de rótulo vazio para {image_file}")
                with open(label_path, "w") as f:
                    pass
                missing_labels.append(image_file)
    return missing_labels

# Função para validar arquivos de rótulo (para YOLOv5)
def validate_labels(label_dir, num_classes=2):
    empty_labels = []
    invalid_labels = []
    valid_labels = []
    if not os.path.exists(label_dir):
        print(f"ERRO: Diretório de rótulos {label_dir} não existe")
        return empty_labels, invalid_labels, valid_labels
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, "r") as f:
                content = f.read().strip()
                if not content:
                    empty_labels.append(label_file)
                    continue
                lines = content.splitlines()
                is_valid = True
                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        invalid_labels.append(f"{label_file}: {line} (Formato inválido, esperado 5 valores)")
                        is_valid = False
                    else:
                        try:
                            class_id, x, y, w, h = map(float, parts)
                            if not (0 <= class_id < num_classes and 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                invalid_labels.append(f"{label_file}: {line} (Valores fora do intervalo)")
                                is_valid = False
                        except ValueError:
                            invalid_labels.append(f"{label_file}: {line} (Valores não numéricos)")
                            is_valid = False
                if is_valid:
                    valid_labels.append(label_file)
    return empty_labels, invalid_labels, valid_labels

# Função para criar um dataset de teste com anotações válidas (para YOLOv5)
def create_test_subset(train_image_dir, train_label_dir, subset_image_dir, subset_label_dir, num_images=5):
    os.makedirs(subset_image_dir, exist_ok=True)
    os.makedirs(subset_label_dir, exist_ok=True)
    if not os.path.exists(train_image_dir):
        print(f"ERRO: Diretório de imagens {train_image_dir} não existe")
        return
    image_files = [f for f in os.listdir(train_image_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        print(f"ERRO: Nenhuma imagem encontrada em {train_image_dir}")
        return
    selected_images = image_files[:min(num_images, len(image_files))]
    for img in selected_images:
        shutil.copy(os.path.join(train_image_dir, img), os.path.join(subset_image_dir, img))
        label_file = os.path.splitext(img)[0] + ".txt"
        subset_label_path = os.path.join(subset_label_dir, label_file)
        with open(subset_label_path, "w") as f:
            f.write("0 0.5 0.5 0.2 0.3\n")  # MOTO: class_id=0, center at (50%, 50%), width=20%, height=30%
        print(f"Criado rótulo de amostra para {img}")

# Criar arquivos de rótulo faltantes (para YOLOv5)
try:
    missing_train_labels = create_missing_labels(train_image_dir, train_label_dir)
    missing_val_labels = create_missing_labels(val_image_dir, val_label_dir)
    print(f"\nImagens sem rótulos (treinamento): {missing_train_labels}")
    print(f"Imagens sem rótulos (validação): {missing_val_labels}")
except FileNotFoundError as e:
    print(f"Erro ao criar rótulos faltantes: {e}")
    raise SystemExit("Por favor, corrija o caminho do diretório e tente novamente.")

# Validar arquivos de rótulo (para YOLOv5)
print("\nValidando rótulos de treinamento...")
empty_train_labels, invalid_train_labels, valid_train_labels = validate_labels(train_label_dir)
print(f"Rótulos vazios (treinamento): {len(empty_train_labels)} ({empty_train_labels})")
print(f"Rótulos inválidos (treinamento): {invalid_train_labels}")
print(f"Rótulos válidos (treinamento): {len(valid_train_labels)} ({valid_train_labels})")
print("\nValidando rótulos de validação...")
empty_val_labels, invalid_val_labels, valid_val_labels = validate_labels(val_label_dir)
print(f"Rótulos vazios (validação): {len(empty_val_labels)} ({empty_val_labels})")
print(f"Rótulos inválidos (validação): {invalid_val_labels}")
print(f"Rótulos válidos (validação): {len(valid_val_labels)} ({valid_val_labels})")

# Criar dataset de teste com anotações válidas (para YOLOv5)
print("\nCriando dataset de teste com anotações válidas...")
create_test_subset(train_image_dir, train_label_dir, subset_train_image_dir, subset_train_label_dir)

# Remover arquivo de cache
cache_path = f"{train_image_dir}.cache"
if os.path.exists(cache_path):
    print(f"Removendo cache: {cache_path}")
    os.remove(cache_path)

# Exibir estrutura do dataset
print("\nEstrutura do dataset original:")
!ls -R {base_dir}
print("\nEstrutura do dataset de teste:")
!ls -R /content/test_dataset

# Passo 2: Configurar a CNN (para alternativa ou complemento ao YOLOv5)
print("\nConfigurando modelo CNN...")
hand_model = Sequential()
hand_model.add(layers.Rescaling(1./255, input_shape=(150, 150, 3)))
hand_model.add(layers.Conv2D(16, kernel_size=10, activation="relu"))
hand_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
hand_model.add(layers.Conv2D(32, kernel_size=5, activation="relu"))
hand_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
hand_model.add(layers.Conv2D(64, kernel_size=3, activation="relu"))
hand_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
hand_model.add(layers.Flatten())
hand_model.add(layers.Dense(128, activation="relu"))
hand_model.add(layers.Dense(2, activation="softmax"))  # 2 classes: MOTO, CACHORRO

hand_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
hand_model.summary()

# Carregar dados para a CNN
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    f"{base_dir}/train",
    target_size=(150, 150),
    batch_size=16,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    f"{base_dir}/val",
    target_size=(150, 150),
    batch_size=16,
    class_mode="categorical"
)

# Treinar a CNN
print("\nTreinando a CNN...")
history = hand_model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator
)

# Passo 3: Instalar YOLOv5
if os.path.exists("/content/yolov5"):
    shutil.rmtree("/content/yolov5")
!git clone https://github.com/ultralytics/yolov5 /content/yolov5
os.chdir('/content/yolov5')
print(f"Diretório de trabalho atual: {os.getcwd()}")

# Instalar dependências
!pip install -r requirements.txt
!pip install albumentations==1.3.1

# Instalar PyTorch CPU
!pip uninstall torch torchvision torchaudio -y
!pip install torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Passo 4: Desativar W&B
os.environ['WANDB_MODE'] = 'disabled'

# Passo 5: Criar arquivo data.yaml (para YOLOv5)
data = {
    'train': subset_train_image_dir,  # Usar subset temporário
    'val': val_image_dir,
    'test': test_image_dir,
    'nc': 2,
    'names': ['MOTO', 'CACHORRO']
}

with open('data.yaml', 'w') as f:
    yaml.dump(data, f)
print("Arquivo data.yaml criado com sucesso!")

# Passo 6: Treinamento YOLOv5
# Simulação 1: 30 épocas
!python train.py --img 640 --batch 16 --epochs 30 --data data.yaml --weights yolov5s.pt --cache --project runs/train --name exp_30epochs

# Simulação 2: 60 épocas
!python train.py --img 640 --batch 16 --epochs 60 --data data.yaml --weights yolov5s.pt --cache --project runs/train --name exp_60epochs

# Passo 7: Validação YOLOv5
!python val.py --weights runs/train/exp_30epochs/weights/best.pt --data data.yaml --img 640 --task val
!python val.py --weights runs/train/exp_60epochs/weights/best.pt --data data.yaml --img 640 --task val

# Passo 8: Teste (Inferência) YOLOv5
!python detect.py --weights runs/train/exp_30epochs/weights/best.pt --img 640 --conf 0.4 --source {test_image_dir} --data data.yaml --save-txt --save-conf --project runs/detect --name exp_30epochs
!python detect.py --weights runs/train/exp_60epochs/weights/best.pt --img 640 --conf 0.4 --source {test_image_dir} --data data.yaml --save-txt --save-conf --project runs/detect --name exp_60epochs

# Passo 9: Exibir resultados
!ls -R runs/detect/exp*
