from flask import Flask, render_template, request, session
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from werkzeug.utils import secure_filename
from glob import glob
from ultralytics import YOLO
import mysql.connector

app = Flask(__name__)
app.secret_key = 'aaa'

# 이전 사용 이미지 있으면 삭제
def clear():
    if os.path.exists("C:/GitTest/Final_Project/runs"):
        shutil.rmtree("C:/GitTest/Final_Project/runs")
    if len(os.listdir('Final_Web/static/input_images/yolo_input')) != 0:
        file_list = glob('Final_Web/static/input_images/yolo_input/*')
        for file_path in file_list:
            os.remove(file_path)

    if len(os.listdir('Final_Web/static/output_images')) != 0:
        file_list = glob('Final_Web/static/output_images/*')
        for file_path in file_list:
            os.remove(file_path)

    if len(os.listdir('Final_Web/static/input_images/car_damage_type/client_input')) != 0:
        file_list = glob(
            'Final_Web/static/input_images/car_damage_type/client_input/*')
        for file_path in file_list:
            os.remove(file_path)

    if len(os.listdir('Final_Web/static/input_images/car_part/part_img')) != 0:
        file_list = glob('Final_Web/static/input_images/car_part/part_img/*')
        for file_path in file_list:
            os.remove(file_path)

# 초기 화면. 이미지 2개로 모델 방법 고르는 페이지.
@app.route('/')
def index():
    clear() # 이전 사용 이미지 있으면 삭제
    return render_template('index.html')


# 로그인 되었을 때만 이용 가능한 기능 회원 아닐경우 경고문구 표시
# 게시판 : 조회 이력 보여주기(미완)
@app.route('/board')
def board():
    return render_template('board.html')

# 버튼 클릭시 이동.
@app.route('/cnn')
def cnn():
    colors = ['그레이', '레드', '블랙', '블루', '실버', '오렌지', '화이트']

    return render_template('upload_files.html', colors=colors)


@app.route('/yolo')
def yolo():
    return render_template('yolo.html')


@app.route('/upload_damage_type', methods=['POST'])
def upload():
    colors = ['그레이', '레드', '블랙', '블루', '실버', '오렌지', '화이트']
    if 'damage_type_image' in request.files:
        uploaded_type_files = request.files.getlist('damage_type_image')
        for uploaded_file in uploaded_type_files:
            # 파일 저장
            filename = secure_filename(uploaded_file.filename)
            file_path = f'Final_Web/static/input_images/car_damage_type/client_input/{filename}'
            uploaded_file.save(file_path)

    if 'part_img_file' in request.files:
        part_img_filename = None
        uploaded_part_file = request.files['part_img_file']
        #filename = secure_filename(uploaded_part_file.filename) ##############################
        new_filename = "part.jpg"
        file_path = f'Final_Web/static/input_images/car_part/part_img/{new_filename}'
        uploaded_part_file.save(file_path)
        part_img_filename = uploaded_part_file

    return render_template('upload_files.html', part_img_filename=part_img_filename, colors=colors)


@app.route('/upload_yolo', methods=['POST'])
def upload2():
    uploaded_file = request.files['yolo_img_file']
    if uploaded_file:
        # 파일 저장
        # filename = uploaded_file.filename
        new_filename = "detect.jpg"
        file_path = f'Final_Web/static/input_images/yolo_input/{new_filename}'
        uploaded_file.save(file_path)
    return render_template('yolo.html')

# YOLO 결과창
# YOLO Bbox그리고 이미지 출력 위해서 복사
@app.route('/yolo_output')
def yolo_output():

    model = YOLO('Final_Web/static/models/best.pt')
    model.predict(source="Final_Web/static/input_images/yolo_input", save=True,
                  conf=0.1, iou=0.02)

    shutil.copy("runs/detect/predict/detect.jpg",
                "Final_Web/static/output_images/detect.jpg")
    return render_template('yolo_output.html')


# CNN 결과창 ------------------------------------------------------------------------------------------------------------------------
@app.route('/out_put', methods=['GET', 'POST'])
def out_put():
    selectedCar = request.form.get('selectedCar')
    selectedYear = request.form.get('selectedYear')
    selectedColor = request.form.get('selectedColor')

    # 파손 부위 모델 예측 함수 호출
    part_output = part_model_predict()
    # 파손 유형 모델 예측 함수 호출
    damage_type_output_arr = damage_type_model_predict()
    # 차량 사이즈 출력 함수 호출

    car_size = car_size_sql(selectedCar)

    # HQ 모델 예측 함수 호출
    # 셀렉트 값 전달 필요 -- 차종만 HQ 모델에 차량 사이즈로 들어간다
    HQ_list = HQ_ML_model(part_output, damage_type_output_arr, car_size)
    # 리스트 3개 받아서 튜플로 저장
    ## HQ_list[0] = HQ_exchange_list
    ## HQ_list[1] = HQ_coating_list
    ## HQ_list[2] = HQ_sheet_metal_list

    # DB에 접속 연산 함수 호출
    exchange_cost_list, sheet_metal_list = DB_HQ_cal(HQ_list, selectedCar, selectedYear, selectedColor, part_output, car_size)

    return render_template('out_put.html', 
                           part_output=part_output, 
                           damage_type_output_arr=damage_type_output_arr, 
                           exchange_cost_list=exchange_cost_list, 
                           sheet_metal_list=sheet_metal_list)

def car_size_sql(car_name):
    # 데이터베이스 연결 설정
    db = mysql.connector.connect(
        host='localhost',
        user='root',
        password='1111',
        database='final_project'
    )
    #'City car', 'Compact car', 'Full-size car', 'Mid-size car', 'SUV', :'VAN'
    car_size_dict = {'소형': 'City car',
                    '경형': 'Compact car',
                    '대형': 'Full-size car',
                    '중형': 'Mid-size car',
                    'SUV': 'SUV',
                    'VAN': 'VAN'}
    
    cursor = db.cursor()
    query = f"SELECT DISTINCT 차량크기 FROM car_part_price WHERE 차이름 = '{car_name}'"  # 차량 이름으로 차량 크기 뽑기
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()

    if result:  # 결과가 있는지 확인
        car_size = car_size_dict.get(result[0][0])  # 첫 번째 요소 참조
        return car_size
    else:
        return None  # 결과가 없을 경우 None 반환


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
    IMAGE_SIZE = 224
    # 이미지 없으면 실행 안함.
    if len(os.listdir('Final_Web/static/input_images/car_part')) == 0:
        return 0

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
    return part_output

# 파손 유형 CNN 함수
def damage_type_model_predict():
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
    # 로딩된 모델 사용 예시 -------------------------------------------------------------------------------------------------------------------------------------------- 예상 문제 구간
    damage_type_output_raw = model.predict(test_generator)  # 상처
    max_index = np.argmax(damage_type_output_raw)
    damage_type_output = damage_type_dict.get(max_index)
    return damage_type_output

# HQ 머신러닝 함수
def HQ_ML_model(part_output, damage_type_output_arr, car_size):

    # 내용 대입전 df 초기화
    #global df
    df = pd.DataFrame(columns=['Bonnet', 'Bumper', 'Door', 'Fender',
                               'Head lights', 'Rear lamp', 'Rocker panel', 'Roof',
                               'Side mirror', 'Trunk lid', 'Wheel',
                               'Breakage', 'Crushed', 'Scratched', 'Separated',
                               'City car', 'Compact car', 'Full-size car', 'Mid-size car', 'SUV', 'VAN']) # 21개. 부위가 Wheel 일경우 도장 생략해야한다.
    # 반복문 위해 damage_type 이미지 개수 카운팅
    img_len = len(os.listdir('Final_Web\static\input_images\car_damage_type\client_input'))
    for i in range(img_len):
        df.loc[i, 0:len(df.columns)] = 0  # 모든열이 0인행 i 추가
        df.loc[i, part_output] = 1  # 파손 부위 열 1로 업데이트
        df.loc[i, damage_type_output_arr] = 1  # 파손 유형 열 1로 업데이트
        df.loc[i, car_size] = 1  # 자동차 사이즈 열 1로 업데이트

    # 로딩된 모델 사용 예시
    # 1. HQ_exchange
    exchange_model = joblib.load(
        'Final_Web/static/models/HQ_exchange_r_forest.pkl')
    exchange_output = exchange_model.predict(df.iloc[:, :21]) # 부위 유형 차종 으로 머신 러닝.

    # 2. HQ_sheet_metal
    sheet_metal_model = joblib.load('Final_Web/static/models/HQ_sheet_metal_r_forest.pkl')
    sheet_metal_output = sheet_metal_model.predict(df.iloc[:, :21])

    # 3. HQ_coating
    if part_output == "Wheel":
        pass
    else:
        del df['Wheel']
        coating_model = joblib.load('Final_Web/static/models/HQ_coating_r_forest.pkl')
        coating_output = coating_model.predict(df.iloc[:, :20])

    for i in range(img_len):
        df.loc[i, 'HQ_exchange'] = exchange_output[i] # 교환 HQ 추가
        df.loc[i, 'HQ_coating'] = coating_output[i] # 도장 HQ 추가
        df.loc[i, 'HQ_sheet_metal'] = sheet_metal_output[i] # 판금 HQ 추가

    HQ_exchange_list = list(df.loc[:, 'HQ_exchange']) # 교환 HQ 전달위해 저장
    HQ_coating_list = list(df.loc[:, 'HQ_coating']) # 도장 HQ 전달위해 저장
    HQ_sheet_metal_list = list(df.loc[:, 'HQ_sheet_metal']) # 판금 HQ 전달위해 저장

    return HQ_exchange_list, HQ_coating_list, HQ_sheet_metal_list

def DB_HQ_cal(HQ_list, selectedCar, selectedYear, selectedColor, part_output, car_size):
    money = 35000
    ## HQ_list[0][0] = HQ_exchange_list
    ## HQ_list[][1] = HQ_coating_list
    ## HQ_list[][2] = HQ_sheet_metal_list
    db = mysql.connector.connect(
        host='localhost',
        user='root',
        password='1111',
        database='final_project'
    )
    cursor = db.cursor()
    # selectedCar = "'" + selectedCar + "'"
    # selectedYear = "'" + selectedYear + "'"
    query = f"SELECT DISTINCT {part_output} FROM car_part_price WHERE \
            차이름 = '{selectedCar}' AND \
            연식 = '{selectedYear}'"  # 차량 이름으로 차량 크기 뽑기
    cursor.execute(query)
    result = cursor.fetchall()
    part_cost = result

    query = f"SELECT DISTINCT {part_output} FROM car_coating_price WHERE \
            차량크기 = '{car_size}'"  # 차량 이름으로 차량 크기 뽑기
    cursor.execute(query)
    result = cursor.fetchall()
    coating_cost = result

    query = f"SELECT price FROM car_color_price WHERE color='{selectedColor}'"
    cursor.execute(query)
    result = cursor.fetchall()
    color_cost = result

    cursor.close()

    exchange_cost_list = []
    sheet_metal_list = []
    

    for i in range(len(HQ_list[0])):
        part_cost_float = float(part_cost[i])
        coating_cost_float = float(coating_cost[i])
        color_cost_float = float(color_cost[i])

        exchange_cost = part_cost_float + money*(HQ_list[i][0]) + money*(HQ_list[i][1]) + coating_cost_float*(HQ_list[i][1])*color_cost_float
        sheet_metal = money*(HQ_list[i][0]) + money*(HQ_list[i][2]) + money*(HQ_list[i][1]) + coating_cost_float*(HQ_list[i][1])*color_cost_float

        exchange_cost_list.append(exchange_cost)
        sheet_metal_list.append(sheet_metal)

    return exchange_cost_list, sheet_metal_list 



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
