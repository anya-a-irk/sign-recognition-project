# sign-recognition-project

## In russian

Проект по распознаванию дорожных знаков в машинке с камерой на Raspberry Pi, Python, OpenCV и keras/TensorFlow.

Подробное описание будет предоставлено в следующей версии.

## Материалы

За основы были взяты и переработаны следующие материалы:

+ [Deep leaning for traffic sign classification in keras and python ](https://github.com/psubnwell/Traffic-Sign-Classification.keras)
+ [Python class for controlling Arduino Motor Shield L293D from Raspberry Pi (using RPi.GPIO) ](https://github.com/neumann-d/AMSpi)
+ Основанный на OpenCV алгоритм выявление дорожного знака на общей картинке, захваченной с камеры
+ [Инструкция по установке собранного пакета OpenCV для Raspberry Pi](https://github.com/jabelone/OpenCV-for-Pi)

## Запуск

`traffic_sign_recognition.py` отвечает за обучение. Выходной файл: `model.h5`. Вместе с кодом из `detect` он записывается на машинку и выполняется. Для обучения использован следующий набор исходных данных: [German Traffic Sign Benchmarks](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

## Выполнили

+ Абдрахимова Анна
+ Котляревская Мария