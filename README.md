#FarmTech Solutions - Sistema de Visão Computacional com YOLOv5
Falha ao carregar a imagemVer link
#Visão Geral
Este repositório contém o projeto de visão computacional desenvolvido para a FarmTech Solutions, focado na detecção de objetos (MOTO e CACHORRO) usando o YOLOv5. O sistema demonstra o potencial da IA para aplicações em segurança patrimonial, saúde animal e outros serviços expandidos da empresa.
O repositório é público para facilitar o acesso pela equipe interna da FIAP, conforme solicitado. Para evitar exposição excessiva, os links para o dataset no Google Drive e o Colab são compartilhados apenas em canais internos ou sob solicitação.
Objetivo
Demonstrar um sistema de detecção de objetos em imagens, com treinamento, validação e teste, destacando a acurácia e escalabilidade do YOLOv5. O projeto segue as metas da entrega 1, incluindo um dataset de 80 imagens, rotulação com Make Sense IA, e comparações de performance com 30 e 60 épocas.
Instalação e Dependências

#Requisitos:

Python 3.8+
Bibliotecas: torch, torchvision, ultralytics/yolov5, albumentations==1.3.1


#Clone o repositório:
textgit clone https://github.com/seu-usuario/farmtech-yolov5.git
cd farmtech-yolov5
pip install -r requirements.txt


#Uso
O código completo, incluindo o passo a passo do treinamento, validação, teste e análise de resultados, está disponível no notebook Colab/Jupyter. Acesse o notebook para executar o sistema e visualizar os achados.
Link para o Colab
Para executar o projeto interativamente:

Acessar o Notebook no Colab (link compartilhado internamente para evitar exposição pública; solicite acesso se necessário).

#No notebook, você encontrará:

Organização do dataset.
Rotulação e configuração.
Treinamento com 30 e 60 épocas.
Validação e teste.
Análise de resultados, com comparações de mAP, loss e performance.

Estrutura do Repositório

data.yaml: Configuração do dataset para YOLOv5.
requirements.txt: Dependências do projeto.
runs/: Pasta gerada com resultados de treinamento, validação e detecção (não versionada, gerada durante execução).
README.md: Esta documentação introdutória.

#Contribuições

#Desenvolvido por
rm562443
rm565433
rm565780
, aluno FIAP.
Baseado em YOLOv5 da Ultralytics.
Dataset de imagens royalty-free do Unsplash.

