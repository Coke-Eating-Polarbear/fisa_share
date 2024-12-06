from blog.models import card
from django.db.models import Q
from collections import defaultdict
import re
import json
from django.db.models import Sum
import os
import pandas as pd
from joblib import load
from django.conf import settings


def card_top(keywords) :
    #card
    benefits = card.objects.values('benefits')

    # Q 객체를 사용하여 OR 조건 생성
    query = Q()
    for keyword in keywords:
        query |= Q(benefits__icontains=keyword)

    # 조건에 맞는 Name 컬럼만 가져오기
    names = card.objects.filter(query).values_list('Name', flat=True)

    # 조건에 맞는 Detail 컬럼 가져오기
    detail = card.objects.filter(query).values_list('Detail', flat=True)

    name_list = list(names)
    detail_list = list(detail)

    # 데이터 초기화
    final_result = defaultdict(dict)  # 키가 없어도 기본값 리스트를 생성

    # 데이터 처리
    for card_name, sentence in zip(name_list, detail_list):
        for keyword in keywords:
            # 키워드와 숫자%가 함께 있는 경우만 추출
            matches = re.findall(rf'{keyword}.*?(\d+%)', sentence)
            if matches:
                # 중복 제거 후 추가
                if keyword in final_result[card_name]:
                    existing = final_result[card_name][keyword]
                    if isinstance(existing, list):
                        final_result[card_name][keyword] = list(set(existing + matches))
                    elif matches[0] not in existing:
                        final_result[card_name][keyword] = [existing] + matches
                else:
                    final_result[card_name][keyword] = matches[0] if len(matches) == 1 else matches

    # 결과 확인
    result_dict = dict(final_result)

    # # 각 카드의 최대 할인값 추출
    # max_discounts = {}

    # for card_name, data in result_dict.items():
    #     discounts = data.get('할인', [])
    #     if isinstance(discounts, str):  # 할인 정보가 문자열이면 리스트로 변환
    #         discounts = [discounts]
    #     if discounts:  # 리스트가 비어 있지 않은 경우만 처리
    #         numeric_discounts = [int(d.replace('%', '')) for d in discounts]
    #         max_discounts[card_name] = max(numeric_discounts)
    #     else:  # 비어 있는 경우 기본값 설정 (예: 0)
    #         max_discounts[card_name] = 0

    # print(max_discounts)

    # %의 숫자를 합산하고 가장 큰 값을 가진 딕셔너리 값 추출

    # 숫자만 추출하고 합산한 값을 계산
    max_card = None
    max_sum = 0

    for card_name, benefits in result_dict.items():
        # 문자열이 아닌 값이 있을 경우 처리
        total_percentage = sum(
            int(value.rstrip('%')) for value in benefits.values() if isinstance(value, str) and value.endswith('%')
        )
        if total_percentage > max_sum:
            max_sum = total_percentage
            max_card_top1 = {card_name: benefits}

    max_card_name = list(max_card_top1.keys())[0]

    max_card_detail_top1 = card.objects.filter(Name=max_card_name).values()

    # eat_max_card_detail_top1_dict = dict(eat_max_card_detail_top1)

    # 리스트를 JSON으로 변환
    max_card_top1_json = json.dumps(max_card_top1, ensure_ascii=False,)
    max_card_datail_top1_json = json.dumps(list(max_card_detail_top1.values()), ensure_ascii=False, indent=4)

    return max_card_top1_json, max_card_datail_top1_json

def amount_category_total(spend_amounts):
    # 각 항목별로 총합을 구합니다.
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
    category_total_dict = {
        '총합': category_totals['total_total_amount'],
    }
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

    # 항목을 값 기준으로 내림차순 정렬하여 상위 7개 항목을 추출
    sorted_categories = sorted(category_dict.items(), key=lambda x: x[1] or 0, reverse=True)

    # 상위 4개 항목을 구합니다.
    sorted_categories = dict(sorted_categories)
    amount_total = dict(category_total_dict)

    # sorted_categories와 amount_total을 JSON으로 변환
    sorted_categories_json = json.dumps(sorted_categories)
    amount_total_json = json.dumps(amount_total)

    # # 나머지 항목을 "기타"로 묶어 총합을 계산 (total 제외)
    # other_categories_total = sum([value for key, value in sorted_categories[7:]])

    # # "기타" 항목 추가
    # top4_categories['기타'] = other_categories_total

    return amount_total_json,sorted_categories_json

def freq_category_total(spend_freq):
    # 각 항목별로 총합을 구합니다.
    Freq_category_totals = spend_freq.aggregate(
        total_eat_Freq=Sum('eat_Freq'),
        total_transfer_Freq=Sum('transfer_Freq'),
        total_utility_Freq=Sum('utility_Freq'),
        total_phone_Freq=Sum('phone_Freq'),
        total_home_Freq=Sum('home_Freq'),
        total_hobby_Freq=Sum('hobby_Freq'),
        total_fashion_Freq=Sum('fashion_Freq'),
        total_party_Freq=Sum('party_Freq'),
        total_allowance_Freq=Sum('allowance_Freq'),
        total_study_Freq=Sum('study_Freq'),
        total_medical_Freq=Sum('medical_Freq'),
        total_total_Freq=Sum('TotalFreq')  # 전체 합계
    )
    # 항목을 한국어로 맵핑한 딕셔너리로 저장
    Freq_category_dict = {
        '식비': Freq_category_totals['total_eat_Freq'] or 0,
        '교통비': Freq_category_totals['total_transfer_Freq'] or 0,
        '공과금': Freq_category_totals['total_utility_Freq'] or 0,
        '통신비': Freq_category_totals['total_phone_Freq'] or 0,
        '주거비': Freq_category_totals['total_home_Freq'] or 0,
        '여가/취미': Freq_category_totals['total_hobby_Freq'] or 0,
        '패션/잡화': Freq_category_totals['total_fashion_Freq'] or 0,
        '모임회비': Freq_category_totals['total_party_Freq'] or 0,
        '경조사': Freq_category_totals['total_allowance_Freq'] or 0,
        '교육비': Freq_category_totals['total_study_Freq'] or 0,
        '의료비': Freq_category_totals['total_medical_Freq'] or 0,
    }

    # 항목을 한국어로 맵핑한 딕셔너리로 저장
    Freq_category_total_dict = {
        '총합': Freq_category_totals['total_total_Freq'],
    }

    # 항목을 값 기준으로 내림차순 정렬하여 상위 7개 항목을 추출
    Freq_sorted_categories = sorted(Freq_category_dict.items(), key=lambda x: x[1] or 0, reverse=True)


    Freq_sorted_categories = dict(Freq_sorted_categories)
    Freq_total = dict(Freq_category_total_dict)

    # sorted_categories와 amount_total을 JSON으로 변환
    Freq_sorted_categories_json = json.dumps(Freq_sorted_categories)
    Freq_total_json = json.dumps(Freq_total)

    return Freq_sorted_categories_json, Freq_total_json


def fetch_sql_processed_data(mydata_pay):
    """
    전처리된 데이터를 만드는 함수.
    Returns:
        DataFrame: SQL에서 처리된 데이터를 Pandas DataFrame으로 반환
    """
    # db_config = {
    #     'host': '118.67.131.22:3306',
    #     'user': 'fisaai',
    #     'password': 'woorifisa3!W',
    #     'database': 'manduck'
    # }
    # db_connection = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
    # engine = create_engine(db_connection)

    # query = """
    # SELECT 
    #     Pyear,
    #     Pmonth,
    #     Bizcode,
    #     SUM(Price) AS TotalPrice,
    #     SUM(SUM(Price)) OVER (PARTITION BY Pyear, Pmonth) AS TotalSpending,
    #     SUM(Price) * 1.0 / SUM(SUM(Price)) OVER (PARTITION BY Pyear, Pmonth) AS Ratio
    # FROM mydata_pay
    # GROUP BY Pyear, Pmonth, Bizcode
    # ORDER BY Pyear, Pmonth, Bizcode;
    # """
    # QuerySet을 Pandas DataFrame으로 변환
    df = pd.DataFrame(list(mydata_pay))
    # df = pd.read_sql(query, engine)
    # 1. TotalPrice 계산: Pyear, Pmonth, Bizcode별로 Price 합산
    df_grouped = df.groupby(['pyear', 'pmonth', 'bizcode'], as_index=False)['price'].sum()
    df_grouped.rename(columns={'price': 'TotalPrice'}, inplace=True)

    # 2. TotalSpending 계산: Pyear, Pmonth별 Price 합산
    df_grouped['TotalSpending'] = df_grouped.groupby(['pyear', 'pmonth'])['TotalPrice'].transform('sum')

    # 3. Ratio 계산: TotalPrice / TotalSpending
    df_grouped['Ratio'] = df_grouped['TotalPrice'] / df_grouped['TotalSpending']

    # 4. 정렬
    df_grouped = df_grouped.sort_values(by=['pyear', 'pmonth', 'bizcode'])


    # 결과 출력
    df=df_grouped

    # Pivot 변환: Bizcode를 열로 만들고 각 Ratio 값을 채움
    pivot_data = df.pivot(index=['pyear', 'pmonth'], columns='bizcode', values='Ratio').fillna(0)

    # TotalSpending 추가
    pivot_data['TotalSpending'] = df.drop_duplicates(subset=['pyear', 'pmonth'])[['pyear', 'pmonth', 'TotalSpending']].set_index(['pyear', 'pmonth'])

    return pivot_data

def predict_next_month(preprocessed_data, model_features):
    """
    가장 최근 데이터를 모델 입력으로 사용하여 다음 달 예측.
    Parameters:
        preprocessed_data (DataFrame): SQL에서 전처리된 데이터
        model_features (list): 모델이 학습된 Bizcode 목록
    Returns:
        Series: 다음 달 예측 결과
    """
    # 가장 최근 데이터 가져오기
    most_recent_period = preprocessed_data.index.max()
    most_recent_data = preprocessed_data.loc[most_recent_period]


    # Series에서 모델 입력 데이터 생성
    model_input = most_recent_data.drop(labels=['TotalSpending'], errors='ignore')

    # 모델 로드 및 예측
    model = load('./models/Consumption_Prediction_rfm.joblib')
    X_test = model_input.values.reshape(1, -1)
    predicted_total = model.predict(X_test)[0]

    # Bizcode별 소비 금액 계산
    predicted_ratios = model_input.values
    predicted_spending = predicted_ratios * predicted_total

    # 결과 반환
    result = pd.Series(
        data=np.append(predicted_spending, predicted_total),
        index=list(model_input.index) + ['predicted_total']
    )
    result.name = (most_recent_period[0], most_recent_period[1] + 1)
    return result

def senter(mydata_pay):
    preprocessed_data = fetch_sql_processed_data(mydata_pay)
    model = os.path.join(settings.BASE_DIR, 'models', 'Consumption_prediction_rfm.joblib')
    model_features = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else preprocessed_data.columns.drop('TotalSpending')

    next_month_prediction = predict_next_month(preprocessed_data, model_features)
    return next_month_prediction


def card_recommend_system(next_month_prediction_json):
    # 식비 관련 키워드
    eat_keywords = ['푸드', '카페', '편의점', '레스토랑', '패밀리레스토랑','배달']
    # 교통비
    transport_keywords = ['대중교통', '교통', '택시', '자동차', '기차', '고속버스', 'SRT', 'KTX']
    # 모임회비
    allowance_keywords = ['용돈', '지원금', '보조금', '수당', '환급', '혜택', '할인']
    # 교육 관련 키워드
    study_keywords = [
        '교육', '학원', '학습', '유치원', '학교', '수업', '강의', '코칭', '레슨', '튜터링',
        '등록금', '학비', '수업료', '수강료', '교육비 지원', '학자금', '장학금',
        '도서', '서적', '온라인 강의', 'E-러닝', '강의 콘텐츠', '교육 콘텐츠', '디지털 학습',
        '교육 프로그램', '학습 도구', '시험', '어학시험', '자격증'
    ]
    # 주거비
    # 주거비 관련 키워드 (공과금 키워드 제외)
    home_keywords = [
        '주거', '임대', '전세', '월세', '매매', '아파트', '빌라', '주택', '부동산',
        '주택자금', '주거비 지원', '대출', '임대료', '보증금', '리모델링'
    ]
    # 공과금 관련 키워드
    utility_keywords = ['전기료', '수도세', '가스비', '관리비', '유지비', '청소비', '공과금']
    # 통신비 관련 키워드
    phone_keywords = ['통신', '이동통신', '전화요금', '인터넷 요금', '휴대폰 요금', '모바일 데이터', '와이파이', '통신비']

    # 여가/취미 관련 키워드
    # 취미/여가 관련 키워드
    hobby_keywords = [
        '영화', '공연', '뮤지컬', '음악', '콘서트', '전시', '미술관', '박물관',
        '테마파크', '여행', '숙박', '캠핑', '글램핑', '낚시', '레저', '스포츠',
        '헬스', '요가', '필라테스', '수영', '등산', '골프', '공연티켓', '놀이공원',
        '액티비티', '도서', '책', '독서', '커뮤니티'
    ]
    # 키워드 필터링
    fashion_keywords = [
        '쇼핑', '온라인쇼핑', '백화점', '베이커리', '패션', '잡화', '의류', '액세서리', '가방', '신발', '구두',
        '뷰티', '화장품', '악세사리', '의류브랜드', '브랜드샵', '패션아이템', '디자인샵', '라이프스타일샵',
        '아울렛', '세일', '할인', '쿠폰', '바우처', '캐시백', '마트/편의점'
    ]
    medical_keywords = [
        '병원', '약국', '병원/약국', '의료', '의료비', '의료기관', '건강관리', '헬스케어',
        '진료비', '건강', '치료', '의료서비스', '클리닉', '재활', '약', '의약품',
        '건강보험', '건강검진'
    ]

    i = 0
    # for문과 if-elif 구조로 연결
    card_results = {}
    card_list = {}
    card_detail_results = {}
    card_list_detail ={}
    max_card_json = None
    max_card_detail_json = None
    if isinstance(next_month_prediction_json, str):
            try:
                next_month_prediction_json = json.loads(next_month_prediction_json)  # 문자열을 딕셔너리로 변환
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}")
                next_month_prediction_json = {}  # 기본값 설정
    for i, keyword in enumerate(top_card_list):
        if keyword == '식비':
            max_card_json, max_card_detail_json = card_top(eat_keywords)
            
            # 값 가져오기
            AmountNum = next_month_prediction_json.get('eat', 0)  # 키가 없으면 기본값 0 반환
        elif keyword == '교통비':
            max_card_json, max_card_detail_json = card_top(transport_keywords)
            # 값 가져오기
            AmountNum = next_month_prediction_json.get('transport', 0)  # 키가 없으면 기본값 0 반환

        elif keyword == '모임회비':
            max_card_json, max_card_detail_json = card_top(allowance_keywords)

            AmountNum = next_month_prediction_json.get('allowance', 0)  # 키가 없으면 기본값 0 반환
        elif keyword == '교육':
            max_card_json, max_card_detail_json = card_top(study_keywords)

            # 값 가져오기
            AmountNum = next_month_prediction_json.get('study', 0)  # 키가 없으면 기본값 0 반환
        elif keyword == '주거비':
            max_card_json, max_card_detail_json = card_top(home_keywords)

            # 값 가져오기
            AmountNum = next_month_prediction_json.get('home', 0)  # 키가 없으면 기본값 0 반환
        elif keyword == '공과금':
            max_card_json, max_card_detail_json = card_top(utility_keywords)
            AmountNum = 0
        elif keyword == '통신비':
            max_card_json, max_card_detail_json = card_top(phone_keywords)

            # 값 가져오기
            AmountNum = next_month_prediction_json.get('phone', 0)  # 키가 없으면 기본값 0 반환
        elif keyword == '여가/취미':
            max_card_json, max_card_detail_json = card_top(hobby_keywords)

            # 값 가져오기
            AmountNum = next_month_prediction_json.get('hobby', 0)  # 키가 없으면 기본값 0 반환
        elif keyword == '패션/잡화/쇼핑':
            max_card_json, max_card_detail_json = card_top(fashion_keywords)

            # 값 가져오기
            AmountNum = next_month_prediction_json.get('fashion', 0)  # 키가 없으면 기본값 0 반환
        elif keyword == '의료':
            max_card_json, max_card_detail_json = card_top(medical_keywords)

            # 값 가져오기
            AmountNum = next_month_prediction_json.get('medical', 0)  # 키가 없으면 기본값 0 반환
        else:
            max_card_json, max_card_detail_json = None, None
            print(f"{keyword}에 해당하는 카테고리가 없습니다.")

        #여기서 할인률, Freq, ammount, discount(할인률 * amount * 0.01)
        AmountNum = round(AmountNum, 2)
        # 할인률
        # max_card_json가 JSON 문자열일 경우 파싱
        if isinstance(max_card_json, str):
            try:
                max_card_json = json.loads(max_card_json)  # JSON 문자열을 딕셔너리로 변환
            except json.JSONDecodeError as e:
                # JSON 파싱 오류 처리
                print(f"JSON decode error: {e}")
                max_card_json = {}  # 파싱 실패 시 기본값 설정
        max_values = {}

        for card_name, benefits in max_card_json.items():
            values = benefits.values()  # 모든 value 값 가져오기
            numeric_values = [int(value.replace('%', '')) for value in values if value.endswith('%')]
            max_values[card_name] = max(numeric_values) if numeric_values else 0  # 최대값 저장

        # 값만 추출
        max_value = list(max_values.values())[0]

        # discount 값
        discount = round(AmountNum * max_value * 0.01, 2)

        # JSON 데이터가 문자열로 되어 있다면, 이를 변환
        if isinstance(max_card_detail_json, str):
            max_card_detail_json = json.loads(max_card_detail_json)

        # 데이터 추가
        if isinstance(max_card_detail_json, list) and max_card_detail_json:
            max_card_detail_json[0]["AmountNum"] = AmountNum
            max_card_detail_json[0]["max_value"] = max_value
            max_card_detail_json[0]["discount"] = discount

        # 필요하면 다시 JSON 문자열로 변환
        max_card_detail_json = json.dumps(max_card_detail_json, ensure_ascii=False)

        # 결과값 저장
        card_results[f"{keyword}"] = max_card_json
        card_detail_results[f"{keyword}"] = max_card_detail_json




    # JSON 문자열 여부를 확인 후 변환
    for key, value in card_detail_results.items():
        if isinstance(value, str):  # value가 JSON 문자열인지 확인
            try:
                card_detail_results[key] = json.loads(value)  # JSON 문자열을 Python 객체로 변환
            except json.JSONDecodeError:
                print(f"Invalid JSON in key {key}: {value}")
        else:
            card_detail_results[key] = value  # 이미 Python 객체라면 그대로 저장

    # `card_results`도 동일한 방식으로 처리
    for key, value in card_results.items():
        if isinstance(value, str):  # value가 JSON 문자열인지 확인
            try:
                card_results[key] = json.loads(value)  # JSON 문자열을 Python 객체로 변환
            except json.JSONDecodeError:
                print(f"Invalid JSON in key {key}: {value}")
        else:
            card_results[key] = value  # 이미 Python 객체라면 그대로 저장

    # JSON으로 변환하여 템플릿에 전달
    card_results_json = json.dumps(card_results, ensure_ascii=False, indent=4)
    card_detail_results_json = json.dumps(card_detail_results, ensure_ascii=False)

    return card_results_json, card_detail_results