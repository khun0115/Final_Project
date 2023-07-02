from flask import Flask, render_template, request, session

app = Flask(__name__)

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
    return render_template('upload_files.html')

# 수정중 ㅁ?ㄹ                                                                   --논의 필요
@app.route('/upload_file')
def upload_file():
    f = request.files['myfile']
    f.save(f.filename)
    return render_template('upload_files.html')


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
    return render_template('out_put.html')

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
            session['id'] = id # 값이 있으면 인증된것이라고 의미, 처리. 보통 아이디
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