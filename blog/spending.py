from django.shortcuts import render
from django.db.models import Sum
from datetime import date
from dateutil.relativedelta import relativedelta
import json
from blog.models import SpendAmount

def calculate_start_date(period, today):
    """기간에 따라 시작 날짜를 계산."""
    start_date = today.month
    if period == '1m':
    # 직전 1달
    # 현재 월에서 한 달을 빼고 그 월의 첫째 날 계산
        if today.month == 1:
            start_date = today.replace(year=today.year - 1, month=12)
        else:
            start_date = today.replace(month=today.month - 1)

    elif period == '6m':
        if today.month <= 6:
            # 1월부터 6월 사이인 경우, 지난 해로 넘어가야 함
            start_date = today.replace(year=today.year - 1, month=12 + (today.month - 6))
        else:
            # 7월 이후의 경우
            start_date = today.replace(month=today.month - 6)
    elif period == '1y':
        # 최근 1년
        start_date = today.replace(year=today.year - 1)
    else:
        # 직전 1달 디폴트
        # 현재 월에서 한 달을 빼고 그 월의 첫째 날 계산
        if today.month == 1:
            start_date = today.replace(year=today.year - 1, month=12)
        else:
            start_date = today.replace(month=today.month - 1)
    
    return start_date  # 기본 1개월

def spend_amount_aggregate(customer_id,start_date):
    spend_amounts = SpendAmount.objects.filter(
    CustomerID=customer_id, 
    SDate__gte=start_date  # 시작 날짜 이후의 데이터만 가져옴
    )
    category_totals = spend_amounts.aggregate(
        total_eat_amount=Sum('eat_amount'),
        total_transfer_amount=Sum('transfer_amount'),
        total_utility_amount=Sum('utility_amount'),
        total_phone_amount=Sum('phone_amount'),
        total_home_amount=Sum('home_amount'),
        total_hobby_amount=Sum('hobby_amount'),
        total_fashion_amount=Sum('fashion_amount'),
        total_party_amount=Sum('party_amount'),
        total_allowance_amount=Sum('allowance_amount'),
        total_study_amount=Sum('study_amount'),
        total_medical_amount=Sum('medical_amount'),
        total_total_amount=Sum('TotalAmount')  # 전체 합계
    )

    # 항목을 한국어로 맵핑한 딕셔너리로 저장
    category_dict = {
        '식비': category_totals['total_eat_amount'] or 0,
        '교통비': category_totals['total_transfer_amount'] or 0,
        '공과금': category_totals['total_utility_amount'] or 0,
        '통신비': category_totals['total_phone_amount'] or 0,
        '주거비': category_totals['total_home_amount'] or 0,
        '여가/취미': category_totals['total_hobby_amount'] or 0,
        '패션/잡화': category_totals['total_fashion_amount'] or 0,
        '모임회비': category_totals['total_party_amount'] or 0,
        '경조사': category_totals['total_allowance_amount'] or 0,
        '교육비': category_totals['total_study_amount'] or 0,
        '의료비': category_totals['total_medical_amount'] or 0,
    }
    return category_totals, category_dict

# 카테고리별 키워드 딕셔너리 정의
KEYWORD_CATEGORIES = {
    '식비': ['푸드', '카페', '편의점', '레스토랑', '패밀리레스토랑', '배달'],
    '교통비': ['대중교통', '교통', '택시', '자동차', '기차', '고속버스', 'SRT', 'KTX'],
    '모임회비': ['용돈', '지원금', '보조금', '수당', '환급', '혜택', '할인'],
    '교육비': [
        '교육', '학원', '학습', '유치원', '학교', '수업', '강의', '코칭', '레슨', '튜터링',
        '등록금', '학비', '수업료', '수강료', '교육비 지원', '학자금', '장학금',
        '도서', '서적', '온라인 강의', 'E-러닝', '강의 콘텐츠', '교육 콘텐츠', '디지털 학습',
        '교육 프로그램', '학습 도구', '시험', '어학시험', '자격증'
    ],
    '주거비': [
        '주거', '임대', '전세', '월세', '매매', '아파트', '빌라', '주택', '부동산',
        '주택자금', '주거비 지원', '대출', '임대료', '보증금', '리모델링'
    ],
    '공과금': ['전기료', '수도세', '가스비', '관리비', '유지비', '청소비', '공과금'],
    '통신비': ['통신', '이동통신', '전화요금', '인터넷 요금', '휴대폰 요금', '모바일 데이터', '와이파이', '통신비'],
    '여가/취미': [
        '영화', '공연', '뮤지컬', '음악', '콘서트', '전시', '미술관', '박물관',
        '테마파크', '여행', '숙박', '캠핑', '글램핑', '낚시', '레저', '스포츠',
        '헬스', '요가', '필라테스', '수영', '등산', '골프', '공연티켓', '놀이공원',
        '액티비티', '도서', '책', '독서', '커뮤니티'
    ],
    '패션/잡화': [
        '쇼핑', '온라인쇼핑', '백화점', '베이커리', '패션', '잡화', '의류', '액세서리', '가방', '신발', '구두',
        '뷰티', '화장품', '악세사리', '의류브랜드', '브랜드샵', '패션아이템', '디자인샵', '라이프스타일샵',
        '아울렛', '세일', '할인', '쿠폰', '바우처', '캐시백', '마트/편의점'
    ],
    '의료비': [
        '병원', '약국', '병원/약국', '의료', '의료비', '의료기관', '건강관리', '헬스케어',
        '진료비', '건강', '치료', '의료서비스', '클리닉', '재활', '약', '의약품',
        '건강보험', '건강검진'
    ]
}

# 특정 카테고리의 키워드를 가져오는 함수
def get_keywords_for_category(category):
    return KEYWORD_CATEGORIES.get(category, [])

# 모든 카테고리와 키워드를 출력하는 함수
def get_all_keywords():
    return {category: keywords for category, keywords in KEYWORD_CATEGORIES.items()}
