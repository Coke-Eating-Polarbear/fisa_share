import os
from copy import deepcopy

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
    
def add_bank_logo(recommend_list, bank_key):
    """
    주어진 추천 리스트에 product_img 키를 추가하여 로고 경로를 삽입합니다.
    """
    updated_list = deepcopy(recommend_list)  # 원본 데이터 보호를 위해 deepcopy
    for item in updated_list:
        bank_name = item.get(bank_key)  # 은행 이름 가져오기
        item['logo'] = get_bank_logo(bank_name)  # 로고 경로 추가
    return updated_list