# multi_modal

# 소스코드 확인
- 소스 코드 실행: run.py
- 데이터 변환: 

## 개별 실험 모델 확인
- 메인 실험모델: experimental > l_model_for_classification_CNN.ipynb
- TS 데이터 제외한 실험 모델: experimental > k_model_for_classification_CNN_nots.ipynb
- MLP mixer 적용 실험 모델: experimental > k_model_for_classification_CNN_catMLP.ipynb


## Using VSCode
1. ssh로 접속합니다(using Remote-ssh plugin).
2. Terminal을 엽니다.
    * Project를 저장할 디렉토리로 갑니다,
    * git clone을 합니다.
    * cd를 사용하여 multi_modal로 접근합니다.
    * poetry shell을 입력
        * 이미 기존에 설치 된 pyenv 환경과 poetry가 설치되어 있어야 합니다.
        ```
        $ poetry shell
        ```
    * poetry install을 입력
        ```
        $ poetry install
        ```
3. VSCode로 돌아와서 커널을 선택합니다.
    * 여기서 주의할 점은, poetry가 만든 .venv 환경에 접근해야 합니다.
        * 만약, 프로젝트 디렉토리 안에 .venv가 없으면 poetry의 option을 변경해줘야 합니다(다른 경로에 있으므로).
            * 해당 명령을 입력합니다.
                ```
                $ poetry config virtualenv.in-project true
                ```
4. 사용하시면 됩니다.

## Using colab
1. https://colab.research.google.com/github/ARP-Lab/multi_modal/blob/main/(file-name).ipynb 로 접근하시면 됩니다.