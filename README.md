# Final_Project

## 팀원소개
[김현욱](https://github.com/khun0115/Final_Project/tree/master/%EA%B9%80%ED%98%84%EC%9A%B1) : 팀장, 데이터 수집, 전처리, DB제작, 통계검증, DL/ML 모델학습

[고은경](https://github.com/khun0115/Final_Project/tree/master/%EA%B3%A0%EC%9D%80%EA%B2%BD) : 데이터 수집, 전처리, 데이터 분석, 통계검증, ML 모델학습

[김도현](https://github.com/khun0115/Final_Project/tree/master/%EA%B9%80%EB%8F%84%ED%98%84) : 데이터 수집, 전처리, 데이터 분석, DL/ML 모델학습, 웹페이지 제작

[김이경](https://github.com/khun0115/Final_Project/tree/master/%EA%B9%80%EC%9D%B4%EA%B2%BD) : 데이터 수집, 전처리, 데이터 분석, 모델학습

[엄진성](https://github.com/khun0115/Final_Project/tree/master/%EC%97%84%EC%A7%84%EC%84%B1) : 데이터 수집, 전처리, 데이터 분석, YOLO/ML 모델학습, 웹페이지 제작

[오원석](https://github.com/khun0115/Final_Project/tree/master/%EC%98%A4%EC%9B%90%EC%84%9D) : 데이터 수집, 전처리, 데이터 분석, DL 모델학습, 발표

[유영익](https://github.com/khun0115/Final_Project/tree/master/%EC%9C%A0%EC%98%81%EC%9D%B5) : 데이터 수집, 전처리, 데이터 분석, DB제작, PPT제작

## 📝Package📝
<a href="https://www.python.org/" target="_blank"><img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/></a>
<a href="https://www.r-project.org/" target="_blank"><img src="https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white"/></a>

[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![ggplot2](https://img.shields.io/badge/ggplot2-FC4E07?style=flat&logo=ggplot2&logoColor=white)](https://ggplot2.tidyverse.org/)
[![PyCharm](https://img.shields.io/badge/PyCharm-000000?style=flat&logo=pycharm&logoColor=white)](https://www.jetbrains.com/pycharm/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-FF6384?style=flat&logo=pytorch&logoColor=white)](https://github.com/ultralytics/yolov5)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)](https://keras.io/)

## 🛠Tools🛠
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/)
[![Jira](https://img.shields.io/badge/Jira-0052CC?style=flat&logo=jira&logoColor=white)](https://www.atlassian.com/software/jira)
[![Synology Chat](https://img.shields.io/badge/%EC%8B%9C%EB%86%80%EB%A1%9C%EC%A7%80%EC%B1%97-4A90E2?style=flat&logo=synology&logoColor=white)](https://www.synology.com/ko-kr/dsm/feature/chat)
[![PowerPoint](https://img.shields.io/badge/PowerPoint-B7472A?style=flat&logo=microsoft-powerpoint&logoColor=white)](https://www.microsoft.com/en-us/microsoft-365/powerpoint)
[![Visual Studio Code](https://img.shields.io/badge/VS%20Code-007ACC?style=flat&logo=visual-studio-code&logoColor=white)](https://code.visualstudio.com/)
<a href="https://jupyter.org/" target="_blank"><img src="https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white"/></a>
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=google-colab&logoColor=white)](https://colab.research.google.com/your-notebook-link)

## 1. 프로젝트 소개
본 프로젝트는 자동차 사고 시 발생하는 다양한 유형의 사고 이미지를 분석하여 손상 유형,
손상 부위 등을 도출하고 이를 기반으로 차량의 예상 수리시간 및  비용을 예측하는 목적을 가지고 있다.
차량 사고 이후 공업사마다 각기 다른 수리비용이 다르기에 정확하게 수리비용을 예측하여 불이익을 당하지 않고자 이번 프로젝트를 진행

## 2. 일정 및 계획
![image](https://github.com/khun0115/Final_Project/assets/127808901/acf709db-86a3-4f07-87aa-2c1651d93f35)
![계획](https://github.com/khun0115/Final_Project/assets/106053306/f976b167-14a2-44db-a0a8-7bde0e55edcd)


## 3. 데이터 출처
차량파손 이미지 : [Ai-Hub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=realm&dataSetSn=581)

차량 부위별 이미지 : [Ai-hub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100)

부품별 가격 데이터 : [HYUNDAI MOBIS](https://www.mobis-as.com/simple_search_part.do)


## 4. 프로젝트 수행 결과


### 2) 통계검증
    - 통계검증
      ANOVA, t검정을 이용하여 변수가 가격에 유의미한 영향을 미치는지 판단하기 위한 검정으로
      모든 변수에 영향을 끼치는걸로 판단.
![image](https://github.com/khun0115/Final_Project/assets/127808901/9443ce5a-bb6a-4e02-b12f-2a04cba02c3c)
![통검](https://github.com/khun0115/Final_Project/assets/106053306/4bfdaf9e-833b-416a-8c64-a7fc32162029)


## 5. 모델 학습


### 1) YOLO
    - 파손 데미지 판별 모델
    데미지별 수리가격이 상이하기에 가격산출을 위해  데미지 손상도를 4가지로 분류
![yolo](https://github.com/khun0115/Final_Project/assets/106053306/4b2348c4-22c4-497e-a9a1-902a4e8dadab)  


      
