# neural-networks-for-software-effort-estimation
Código desarrollado en python, como base de la tesis: "Desarrollo y validacion de modelos de redes neuronales para evaluar la estimación de esfuerzos en proyectos de desarrollo de software

# NNEffortEstimation

Este proyecto implementa redes neuronales para la estimación de esfuerzo en proyectos de desarrollo de software, usando arquitecturas MLP y el modelo híbrido CNN-GRU.

## Requisitos

Para instalar las dependencias:

pip install -r requirements.txt

## Estructura del Proyecto

- `src/`: Código fuente principal
- `data/`: Datasets en formato ARFF
- `models/`: Modelos entrenados y guardados
- `results/`: Métricas y gráficas resultantes

## Uso
bash
cd src
python NNEffortEstimation.py --datasets albrecht china desharnais --models mlp cnn_gru

### Argumentos

- `--datasets`: Lista de datasets a utilizar (opcional, por defecto: todos)
- `--models`: Modelos a entrenar (opciones: mlp, cnn_gru)
Si no se indican argumentos, se entrenan los dos modelos con todos los datasets.

## Resultados

Los resultados se almacenan en:
- Métricas: `results/metrics/`
- Gráficas: `results/plots/`
- Modelos: `models/saved_models/`
