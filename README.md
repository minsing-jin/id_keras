# id_keras
- 개발환경설치
1. 아나콘다설치 -> python 3.8.8
2. 그래픽 드라이버 설치 -> nvidia-smi -> 466.27
3. cuda 11.0 설치 (visual studio intigration 체크 제거)(nvcc -V)->버전확인
4. CuDNN 8.0.5 for Cuda 11.0 설치
5. tensorflow 설치
        pip install tensorflow-gpu==2.4.1
        pip install tensorflow==2.4.1
        pip install tensorflow-cpu==2.4.1
        
        삭제: uninstall tensorflow-gpu
