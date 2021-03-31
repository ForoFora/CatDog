# CatOrDog
Учебный проект использующий нейронную сеть InceptionV3 для реализации определения Котов и Собак по изображению.
![CatOrDog](https://github.com/ForoFora/ForoFora/blob/main/img/catordog.jpg?raw=true)

## Transfer learning.
Основной смысл transfer learning. Архитектура deeplearning (а точнее сверточных нейросетей, в частности inception) похожа на работу зрительной коры мозга.
Она состоит из нескольких уровней, каждый из которых определяет всё более сложные формы.
На нижнем (ближайшем к глазу / входному изображению) сеть распознаёт различные ключевые точки, на следующем — выстраивает из них линии, затем более сложные фигуры, и так далее.
И лишь на последних уровнях по целой кучи характерных паттернов происходит определение объекта.
Чтобы обучить нейростеть такой сложности необходимо гигантское количество примеров (порядок — от десятков тысяч до миллионов).
Transfer learning вместо обучения всей сети целиком предлагает дообучить лишь верхние слои, по которым происходит непосредственно определение объектов.
Это работает, так как у большинство объектов похоже на уровне основных визуальных паттернов — почти везде будут различные линии (разной степени изогнутости), градиенты, выпуклости, острые части, сочетания цветов и прочие.
Для transfer-learning-а достаточно коллекции из сотен / тысяч элементов.

## Установка
Для сервера web-приложения, следует установить **[Node.JS LTC](https://nodejs.org/en/)** (17.03.2021)  
После установки произвести установку пакетов требуемых для проекта
```
npm install
```

Для сервера обработки, требуется python версии **[Python 3.7.*](https://www.python.org/downloads/release/python-379/)** (17.03.2021)   
После установки развернем и активируем виртуальное окружение в папке проекта
```
python -m venv --system-site-packages .\venv

.\venv\Scripts\activate
```

Устанвливаем следующие пакеты для работы с нейронной сетью (в активированном виртуальном окружении)
```
pip install --upgrade tensorflow
pip install pillow
pip install numpy
pip install keras
pip install flask
```

## Подготовка к запуску
1. Разархивировать папку **data** хранящую тренировочные и валидационные данные
2. python train_1.py
3. python train_2.py
4. python train_3.py
5. python serverPy_4.py*
6. node server.js
7. Перейти по адресу http://localhost:3000/

## Описание шагов
2. Создание numpy массивов для обучения верхнего слоя сети
3. Создание и обучение верхнего слоя (2 слоя по 64 нейрона)
4. Объединение верхнего и нижних слоев модели, обучение общей модели
5. Создание flask сервера, с обученной моделью
```
*В файле serverPy_4.py найдите строчку (17) типа:

weights_filename='new_model_weights/weights-improvement-02-0.88.hdf5'

Вставьте свое имя файла, которое получила ваша модель на 4м шаге в папке new_model_weights
Например: weights-improvement-03-0.90.hdf5
```
6. Создание сервера web-приложения

## Установка библиотек в Azure DevOps
Список команд:
1) sudo su

2) apt update

3) apt install git

4) git version

5) cd /usr/

6) git clone https://github.com/ForoFora/CatDog.git

7) cd CatDog

8) curl -sL https://deb.nodesource.com/setup_15.x | sudo bash -

9) sudo apt install nodejs

10) apt install python3.8

11) alias python=python3.8

12) apt install python3-pip

13) python -m pip install pip

14) update-alternatives --remove python /usr/bin/python2

15) update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10

16) python -m pip install --upgrade pip

17) pip install --upgrade tensorflow

18) pip install pillow

19) pip install numpy

20) pip install keras

21) pip install flask

22) apt install unrar

23) unrar x data.rar

24) python train_1.py 

25) python train_2.py

26) python train_3.py

27) cd new_model_weights/

28) ls -la

29) cd ..

30) nano serverPy_4.py

31) cat serverPy_4.py

32) python serverPy_4.py

33) Ctrl + С

34) apt install node-gyp

35) apt install npm

36) npm install

37) nano server.js

38) cat server.js

39) node server.js

40) Ctrl + С

41) cd /etc/systemd/system

42) touch catOrDogFront.service

43) nano catOrDogFront.service
```
[Unit]
Description=Node JS server for CatDog project
After=network.target

[Service]
User=root
TimeoutSec=0
WorkingDirectory=/usr/CatDog/
ExecStart=/usr/bin/node server.js
KillMode=process

Restart=on-failure
RestartSec=42s

[Install]
WantedBy=default.target
```

44) touch catOrDogNN.service

45) nano catOrDogNN.service
```
[Unit]
Description=Neural network for CatDog project
After=network.target

[Service]
User=root
TimeoutSec=0
WorkingDirectory=/usr/CatDog/
ExecStart=/usr/bin/python serverPy_4.py
KillMode=process

Restart=on-failure
RestartSec=42s

[Install]
WantedBy=default.target
```
46) systemctl start catOrDogFront.service

47) systemctl status catOrDogFront.service

48) systemctl start catOrDogNN.service

49) systemctl status catOrDogNN.service
