import os

def get_bank_logo(bank_name):
    """
    주어진 은행 이름에 해당하는 로고 이미지 파일명을 반환합니다.
    """
    logo_filename = f"{bank_name}.PNG"
    logo_path = os.path.join("img/", logo_filename)  # 로고 파일이 저장된 디렉토리

    # 파일 존재 여부 확인
    if os.path.exists(os.path.join("static", logo_path)):
        return logo_path
    else:
        return "img/default_logo.png"