# multi_modal

## 1. Prerequisite

### 1-1. Dependency 설정
* poetry shell을 입력
* 이미 기존에 설치 된 pyenv 환경과 poetry가 설치되어 있어야 합니다.
    ```
    $ poetry shell
    ```
* poetry install을 입력
    ```
    $ poetry install
    ```

### 1-2. .env 생성
* 본 project는 .env를 Project root에 생성해야 하며, .env 파일에는 다음과 같은 내용이 있어야 합니다.
    ```
    uni_nn_type=torch
    zconf_path=./model/conf
    wandb_keys=
    ```
* 다음과 같은 옵션이 포함될 수 있습니다.
    - uni_nn_type : UniversalNN을 동작시킬 때 선택하는 NN Framework(여기서는 torch).
    - zconf_path : 각 모델에 Configuration을 하기 위해 사용하는 conf file의 집합의 Directory Path.
    - wandb_keys : wandb의 key를 입력(string)

## 2. Source

### 2-1. 실행
- 소스 코드 전체 실행
    ```
    python run.py --all
    ```
- 데이터 생성(Data Pre-Processing ~ Make Data)
    ```
    python run.py --make_data
    ```
- Jupyter Notebook으로 전체 과정 실행
    - run.ipynb

### 2-2. 개별 실험 모델 확인(실험 및 실험 결과)
- 메인 실험모델
    - experimental/l_model_for_classification_CNN.ipynb
- TS 데이터 제외한 실험 모델
    - experimental/k_model_for_classification_CNN_nots.ipynb
- MLP mixer 적용 실험 모델
    - experimental/k_model_for_classification_CNN_catMLP.ipynb

## 3. 사용법

### 3-1. Using VSCode
1. ssh로 접속합니다(using Remote-ssh plugin).
2. Terminal을 엽니다.
    * Project를 저장할 디렉토리로 갑니다,
    * git clone을 합니다.
    * cd를 사용하여 multi_modal로 접근합니다.
    * 1-1의 dependency 설정에 따릅니다.
    
3. VSCode로 돌아와서 커널을 선택합니다.
    * 여기서 주의할 점은, poetry가 만든 .venv 환경에 접근해야 합니다.
        * 만약, 프로젝트 디렉토리 안에 .venv가 없으면 poetry의 option을 변경해줘야 합니다(다른 경로에 있으므로).
            * 해당 명령을 입력합니다.
                ```
                $ poetry config virtualenv.in-project true
                ```
4. 사용하시면 됩니다.

### 3-2. Using colab
1. https://colab.research.google.com/github/ARP-Lab/multi_modal/blob/main/run.ipynb 로 접근하시면 됩니다.

# Authors
- Louan Lee (feelinppl@gmail.com)
- Doohoon Kim (invi.dh.kim@gmail.com)
- Kyungho Kim (khk172216@gmail.com)
- Sooik Jo (show454@naver.com)
- Geuncheol Oh
- Juwon Kim

# License
Copyright 2023. ARP Lab(Louan Lee, Doohoon Kim, Kyungho Kim, Sooik Jo, Guenchul Oh, Juwon Kim) all rights reserved.