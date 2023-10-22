<h1 align="center"><b><u>NNFS</u></b></h1>


<h3 align="center">Convolutional Neural Network Framework from scratch</h3>

<div align="center">
    
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)

</div>
<div align="center">
    
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pUW5rRX4Z_DEVDFLZi-90cBYw-38Rp_a?usp=sharing]

</div>

---

## 💼 Фреймворк для построения свёрточных нейронных сетей, реализованный с нуля

Данный проект реализован в рамках *"челленджа"* по предмету "Вычислительная математика" при обучении на направлении "Информатика и Вычислительная техника" на Космическом факультете в МФ МГТУ им. Н. Э. Баумана 
*(куратор проекта - Малашин А. А.)*

### ⚡️ Реализованные компоненты
В проекте реализованные следующие компоненты, позволяющие сформировать модель нейронной сети практически под любые нужды:

1. **Слои**

    - `Layer_Dense` Плотный слой
    - `Convolutional` Слой свёртки
    - `Layer_Dropout` Dropout-слой
    - `Reshape` Слой для изменения размерности данных

2. **Функции активации**

    - `Activation_ReLU` на базе ReLU-функции
    - `Activation_Softmax` на базе Softmax-функции для результирующего слоя
    - `Activation_Sigmoid` на базе сигмоиды
    - `Activation_Linear` на базе линейной зависимости (для примера)

3. **Оптимизаторы**

    - `Optimizer_SGD` SGD оптимизаторов
    - `Optimizer_Adagrad` Adagrad оптимизатор 
    - `Optimizer_RMSprop` оптимизатор RMSprop
    - `Optimizer_Adam` Adaptive Moment оптимизатор (ультимативное решение)

4. **Функции потери**

    - `Loss_CategoricalCrossentropy` используется при построении модели с задачами классификации
    - `Loss_BinaryCrossentropy` используется при построении модели с задачами классификации
    - `Loss_MeanSquaredError` используется для регрессионных моделей
    - `Loss_MeanAbsoluteError` используется для регрессионных моделей

5. **Подсчет точности**

    - `Accuracy_Categorical` используется при построении модели с задачами классификации
    - `Accuracy_Regression` используется для регрессионных моделей

6. **Класс-интерфейс модели**

    `Model` позволяет добавлять слои в модель, производить обучение и валидацию с учетом заданных функций потерь, точности и оптимизаторов

---

## 🖥️ Пример использования

> ⚠ Актуальный подробный пример описан в `test.py файле`

Create `Model` object
```python
import Model from NNFS_DIY
model = Model()

model.add(...)
...

model.set(
    loss=*Loss Class*,
    optimizer=*Optimizer Class*,
    accuracy=*Accuracy Class*
)

model.finalize()

model.train(*Input Data*, *Output Data*,
            validation_data=(*Test Input*, *Test Output*), 
            epochs=*Epochs*, 
            batch_size=*Items in batch*,
            print_every=*No of epoch*)

```

## 📑 Использованные материалы

При реализации студенты опирались на информацию и примеры кода, описанные в книге [Neural Networks from Scratch in Python](https://nnfs.io)

## ℹ️ Дополнительные ссылки

- [Ссылка на Google Collab с конспектами по проекту](https://colab.research.google.com/drive/1pUW5rRX4Z_DEVDFLZi-90cBYw-38Rp_a?usp=sharing)
- [Ссылка на презентацию проекта на Gamma.app](https://gamma.app/docs/--hpetfyeq230datm?mode=present#card-tm3tbwtxcoj6zwh)

> Если вам понравились материалы проекта, оставьте, пожалуйста, звездочку этому репозиторию ⭐️
