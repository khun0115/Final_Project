from flask import Flask, render_template, request, session
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Flatten, Input, concatenate
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

UPLOAD_FOLDER = '/path/to/save/folder'  # 저장할 폴더 경로
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 초기 화면 보여주기. 이미지 2개로 모델 방법 고르는 페이지.
# accordian - index 계승 : 하단에 사이트 이용법 설명


@app.route('/')
def index():
    return render_template('index.html')


# 로그인 되었을 때만 이용 가능한 기능 회원 아닐경우 경고문구 표시
# 게시판 : 조회 이력 보여주기
@app.route('/board')
def board():
    return render_template('board.html')

# 이미지 혹은 버튼(추가에정?) 클릭시 이동.
# 부위사진 1장과 상처별로 사진 1장씩 입력받아서 cnn돌리기 위함.


@app.route('/cnn')
def cnn():
    # f = request.files['part_img_file']
    # f.save(f.filename)

    colors = ['그레이', '레드', '블랙', '블루', '실버', '오렌지', '화이트']
    return render_template('upload_files.html', colors=colors)


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'part_img_file' not in request.files:
#         return 'No file part_img'

#     file = request.files['part_img_file']
#     if file.filename == '':
#         return 'No selected file'

#     # 저장할 파일 경로 생성
#     save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

#     # 이미지 저장
#     file.save(save_path)

#     return 'File uploaded successfully'

@app.route('/upload_damage_type', methods=['POST'])
def upload():
    # 업로드한 파일 가져오기
    uploaded_file = request.files['damage_type_image']

    if uploaded_file:
        # 파일 저장
        filename = uploaded_file.filename
        uploaded_file.save(f'Final_Web/static/input_images/car_damage_type/client_input/{filename}')

        return '파일 업로드 완료'

    return '파일 업로드 실패'


# 수정중 ㅁ?ㄹ                                  --논의 필요
# @app.route('/upload_file')
# def upload_file():
#     f = request.files['myfile']
#     f.save(f.filename)

#     colors = ['그레이', '레드', '블랙', '블루', '실버', '오렌지', '화이트']
#     return render_template('upload_files.html', colors=colors)


# 결과창 공통점
# 1. 이미지 상단에 선택한 차종등 정보 제목으로 출력
# 2. 입력받은 정보들 저장된 모델 불러와 적용


# 3-1. YOLO는 Predict된 유형들 Bbox 그려진 이미지 출력.
# 상처 감지한 개수 만큼 반복문 => 상처당 수리 방법에 따른 금액 3가지 출력

# 3-2. CNN은 사진 보여줄까?말까?                                                --논의 필요
# 상처 사진 입력받은 개수 만큼 반복문 => 상처당 수리 방법에 따른 금액 3가지 출력
#
@app.route('/out_put', methods=['GET', 'POST'])
def out_put():
    # 업로드된 파일들을 저장할 경로
    # upload_folder = '/static/input_images'

    # # 업로드된 파일들을 받아서 저장
    # if not os.path.exists(upload_folder):
    #     os.makedirs(upload_folder)

    # for file in request.files.getlist('image'):
    #     if file:
    #         file.save(os.path.join(upload_folder, file.filename))

    # 부위모델 예측
    #part_img = 'static/input_images/car_part/*'
    part_output = part_model_predict()

    # 파손 유형 모델 예측
    #type_img_list = 'static/input_images/car_damage_type/*'
    damage_type_output = damage_type_model_predict()

    # 입력받은 값들로 X요소 추출
    # 차종 => car_size

    # car_kind = selectbox.step3.values --??????

    # HQ 모델 예측
    hq_output = HQ_ML_model(part_output, damage_type_output)

    # , hq_output
    return render_template('out_put.html', part_output=part_output, damage_type_output=damage_type_output, hq_output=hq_output)


def part_model_predict():
    # 이미지들로 model.evaluate()
    car_part_dict = {0: 'Bonnet',
                     1: 'Bumper',
                     2: 'Door',
                     3: 'Fender',
                     4: 'Head lights',
                     5: 'Rear lamp',
                     6: 'Rocker panel',
                     7: 'Roof',
                     8: 'Side mirror',
                     9: 'Trunk lid',
                     10: 'Wheel'}
    #
    IMAGE_SIZE = 224
    print(os.getcwd())
    if len(os.listdir('Final_Web/static/input_images/car_part')) == 0:
        return 0
    else:
        pass

    test_data_gen = ImageDataGenerator(rescale=1./255)
    test_generator = test_data_gen.flow_from_directory(directory=f'Final_Web/static/input_images/car_part',
                                                       color_mode='rgb',
                                                       target_size=(
                                                           IMAGE_SIZE, IMAGE_SIZE),
                                                       class_mode='sparse')

    model = tf.keras.models.load_model('Final_Web/static/models/car_part.h5')
    # 로딩된 모델 사용 예시
    part_output_raw = model.predict(test_generator)
    max_index = np.argmax(part_output_raw)
    part_output = car_part_dict.get(max_index)
    # part_output = {'chr_type': car_part_dict.get(max_index) , "int_type": max_index}
    return part_output


def damage_type_model_predict():
    # 이미지들로 model.evaluate()
    damage_type_dict = {0: 'Breakage',
                        1: 'Crushed',
                        2: 'Scratched',
                        3: 'Separated'}
    #
    IMAGE_SIZE = 224
    test_data_gen = ImageDataGenerator(rescale=1./255)
    test_generator = test_data_gen.flow_from_directory(directory=f'Final_Web/static/input_images/car_damage_type',
                                                       color_mode='rgb',
                                                       target_size=(
                                                           IMAGE_SIZE, IMAGE_SIZE),
                                                       class_mode='sparse')

    model = tf.keras.models.load_model(
        'Final_Web/static/models/car_damage_type.h5')
    # 로딩된 모델 사용 예시
    damage_type_output_raw = model.predict(test_generator)  # 상처
    max_index = np.argmax(damage_type_output_raw)
    damage_type_output = damage_type_dict.get(max_index)
    return damage_type_output


def HQ_ML_model(part_output, damage_type_output):
    model = joblib.load(
        'Final_Web/static/models/HQ_exchange_r_forest.pkl')
    # 내용 대입전 df 초기화
    df = pd.DataFrame(columns=['Bonnet', 'Bumper', 'Door', 'Fender',
                               'Head lights', 'Rear lamp', 'Rocker panel', 'Roof',
                               'Side mirror', 'Trunk lid', 'Wheel',
                               'Breakage', 'Crushed', 'Scratched', 'Separated',
                               'City car', 'Compact car', 'Full-size car', 'Mid-size car', 'SUV', 'VAN'])
    df.loc[len(df)] = 0  # 모든행이 0인행 1개 추가
    df[part_output] = 1  # 파손 부위 열 1로 업데이트
    df[damage_type_output] = 1  # 파손 유형 열 1로 업데이트

    # 로딩된 모델 사용 예시

    hq_output = model.predict(df)
    return hq_output

# 로그인 회원만 사용 할 수 있는 기능. 회원 아니면 회원 전용 기능 disable
# 이전 조회 기록 조회.
# 메인에 이름 표시로 바뀜


@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == "GET":
        return render_template('login.html')
    else:
        id = request.form['username']
        pw = request.form['userpass']

        if id == 'tiger' and pw == "1111":
            session['id'] = id  # 값이 있으면 인증된것이라고 의미, 처리. 보통 아이디
            return render_template('index.html')
        else:
            return render_template('login.html', msg=True)

# 로그 아웃


@app.route('/logout')
def logout():
    # session.clear()
    session['id'] = False
    return render_template('index.html')


if __name__ == "__main__":
    app.secret_key = '1234'
    app.run(port=8080, host="0.0.0.0", debug=True)
