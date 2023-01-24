from deepface import DeepFace
import json

##### ФУНКЦИЯ ДЛЯ ПОЛУЧЕНИЯ ИНФОРМАЦИИ О ЧЕЛОВЕКЕ ПО ЕГО ФОТО #####

# img - путь до изображения

# list - категории по которым
# будет анализироваться фотография

# блок try-except для обработки ошибок

# DeepFace.analyze() -
# функция для получения
# информации(возраст, пол) о
# человеке по его фотографии

# img_path, actions - параметры функции analyze()

# info.json - файл формата JSON куда мы записываем
# полученный результат информации о человеке на фото


def photo_info(img, list):
    try:
        result_analyze = DeepFace.analyze(img_path=img, actions=list)
        with open('info.json', 'w') as file:
            json.dump(result_analyze, file, indent=4, ensure_ascii=False)

        #print('Результат анализа \n', result_analyze)

        print(f'[+] Age: {result_analyze.get("age")}')

        print(f'[+] Gender: {result_analyze.get("gender")}')

        print('[+] Race:')
        for k, v in result_analyze.get("race").items():
            print(f'{k} - {round(v, 2)}%')

        print('[+] Emotions:')
        for k, v in result_analyze.get("emotion").items():
            print(f'{k} - {round(v, 2)}%')


    except Exception as _ex:
        return _ex


photo_info(img='D:\\Diplom\\TestNeuralNetworks\\photo\\scarlet.jpg', list=['age', 'gender', 'emotion', 'race'])