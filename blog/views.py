from django.shortcuts import render, redirect# type: ignore
from django.contrib.auth import logout # type: ignore
from django.utils import timezone # type: ignore
from datetime import timedelta
from datetime import date
import base64
from matplotlib import rc # type: ignore
from blog.models import UserProfile,Recommend, Wc, News, Favorite, Average,card, MyDataAsset, MyDataDS, MyDataPay,SpendAmount, DProduct, SProduct, SpendFreq  # UserProfile 모델도 가져옵니다
from django.contrib.auth.hashers import check_password# type: ignore
from django.views.decorators.http import require_POST# type: ignore
from django.http import JsonResponse# type: ignore
from django.db.models import F # type: ignore
import logging
from .logging import *
from elasticsearch import Elasticsearch # type: ignore
from django.views.decorators.csrf import csrf_exempt # type: ignore
import json
import os
from dotenv import load_dotenv # type: ignore
from accounts.views import map_person
import pandas as pd
from datetime import datetime
from joblib import load
import numpy as np
from django.conf import settings
from openai import OpenAI
from django.db.models import Q
from django.core.serializers.json import DjangoJSONEncoder
from django.core import serializers
from django.db.models import Sum
from dateutil.relativedelta import relativedelta
from django.db.models import Q
import re
from collections import Counter
import requests
from collections import defaultdict
from copy import deepcopy
es = Elasticsearch([os.getenv('ES')])  # Elasticsearch 설정
load_dotenv() 
# openai.api_key = os.getenv('APIKEY')
client = OpenAI()
rc('font', family='Malgun Gothic')
logger = logging.getLogger(__name__)

def get_sorted_categories_json(customer_id, start_date):
    """
    특정 고객의 소비 데이터를 기반으로 sorted_categories_json을 생성합니다.
    """
    # `SpendAmount`에서 기간에 맞는 데이터 필터링
    spend_amounts = SpendAmount.objects.filter(
        CustomerID=customer_id, 
        SDate__gte=start_date  # 시작 날짜 이후의 데이터만 가져옴
    )
    
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

    # 항목을 값 기준으로 내림차순 정렬하여 상위 항목 추출
    sorted_categories = sorted(category_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_categories_dict = dict(sorted_categories)

    # JSON 형식으로 변환
    sorted_categories_json = json.dumps(sorted_categories_dict)
    
    return sorted_categories_json

def login_required_session(view_func):
    """
    세션에 'user_id'가 없을 경우 로그인 페이지로 리디렉션하는 데코레이터.
    """
    def wrapper(request, *args, **kwargs):
        # 세션에 'user_id'가 없으면 로그인 페이지로 리디렉션
        if not request.session.get('user_id'):
            return redirect('accounts:login')  # 'login' URL로 이동
        # 'user_id'가 있으면 원래의 뷰 함수 실행
        return view_func(request, *args, **kwargs)
    return wrapper

def logout_view(request):
    # 모든 세션 초기화
    logout(request)
    
    # main.html로 리다이렉트
    return redirect('main')  # 'main'은 urls.py에서 정의된 main.html의 URL name

@login_required_session
def update_profile(request):
    # 세션에서 user_id 가져오기
    user_id = request.session.get('user_id')
    if not user_id:
        return redirect('login')  # 로그인 페이지로 리디렉션

    # 데이터베이스에서 회원 정보 가져오기
    user = UserProfile.objects.get(CustomerID=user_id)
    birth_year = user.Birth.year
    current_year = datetime.now().year
    age = current_year - birth_year

    # 역매핑 계산
    reverse_data = reverse_mapping_with_age(user.Stageclass, age)

    if request.method == 'POST':
        # 사용자 입력 데이터 가져오기
        username = request.POST['username']
        pw = request.POST['Pw']
        email = request.POST['Email']
        phone = request.POST['Phone']

        # 결혼 여부, 자녀 여부, 자녀 나이대
        marital_status = request.POST['marital_status']
        children_status = request.POST.get('children_status') == 'Y'
        children_age = request.POST.get('children_age')

        # StageClass 다시 매핑
        updated_stage_class = map_person(age, marital_status, children_status, children_age)

        # 데이터 업데이트
        user.username = username
        user.Pw = pw
        user.Email = email
        user.Phone = phone
        user.Stageclass = updated_stage_class
        user.save()

        return redirect('mypage')  # 프로필 페이지로 리디렉션

    # GET 요청 시 회원 정보 렌더링
    context = {
        'user': user,
        'reverse_data': reverse_data,
    }
    return render(request, 'update_profile.html', context)

def reverse_mapping_with_age(category, age):
    if category == 'A':
        return {'marriage_status': 'N', 'children_status': 'N', 'children_age': None, 'age_range': '20대'}
    elif category == 'B':
        if 30 <= age < 40:
            return {'marriage_status': 'N', 'children_status': 'N', 'children_age': None, 'age_range': '30대'}
        elif 40 <= age < 50:
            return {'marriage_status': 'N', 'children_status': 'N', 'children_age': None, 'age_range': '40대'}
    elif category == 'C':
        return {'marriage_status': 'Y', 'children_status': 'N', 'children_age': None, 'age_range': '20~40대'}
    elif category == 'D':
        return {'marriage_status': 'Y', 'children_status': 'Y', 'children_age': '초등생', 'age_range': '20~30대'}
    elif category == 'E':
        return {'marriage_status': 'Y', 'children_status': 'Y', 'children_age': '초등생', 'age_range': '40대'}
    elif category == 'F':
        return {'marriage_status': 'Y', 'children_status': 'Y', 'children_age': '중고등생', 'age_range': '40대'}
    elif category == 'G':
        return {'marriage_status': 'Y', 'children_status': 'Y', 'children_age': '중고등생', 'age_range': '50대'}
    elif category == 'H':
        return {'marriage_status': 'Y', 'children_status': 'Y', 'children_age': '성인자녀', 'age_range': '50대'}
    elif category == 'I':
        if age >= 60:
            return {'marriage_status': None, 'children_status': None, 'children_age': None, 'age_range': '60대'}

@login_required_session
def mypage(request):
    customer_id = request.session.get('user_id')  # 세션에서 CustomerID 가져오기
    user_name = "사용자"
    accounts = []  # 사용자 계좌 정보 저장
    expiring_accounts = []  # 만기일이 90일 이내로 남은 계좌
    expiring_accounts_json = None
    accounts_list, d_list, s_list, mypay = [], [], [], []
    total_spent, goal_amount, comparison = 0, None, None  # 지출 금액, 목표 금액, 비교 결과 초기화

    if customer_id:
        try:
            # CustomerID로 UserProfile 조회
            user = UserProfile.objects.get(CustomerID=customer_id)
            user_name = user.username  # 사용자 이름 설정
            category_totals = defaultdict(int)
            # MyDataDS 모델에서 해당 CustomerID에 연결된 계좌 정보 가져오기
            accounts = MyDataDS.objects.filter(CustomerID=customer_id).values('AccountID', 'balance','pname', 'ds_rate','end_date','dstype')
            # 오늘 날짜 계산
            today = timezone.now().date()
            now = datetime.now()
            current_year = now.year
            current_month = now.month
            mypay = MyDataPay.objects.filter(
                CustomerID=customer_id,
                pyear=current_year,
                pmonth=current_month
            ).values('pdate', 'bizcode', 'price', 'pyear', 'pmonth')

            # 목표 금액 가져오기
            goal_amount = user.goal_amount

            category_mapping = {
                'eat': '식비',
                'transfer': '교통비',
                'utility': '공과금',
                'phone': '통신비',
                'home': '주거비',
                'hobby': '여가/취미',
                'fashion': '패션/잡화',
                'party': '모임회비',
                'allowance': '경조사',
                'study': '교육비',
                'medical': '의료비',
            }
            for item in mypay:
                if item['bizcode'] in category_mapping:
                    category_totals[category_mapping[item['bizcode']]] += item['price']
            # 총 지출 계산
            total_spent = sum(item['price'] for item in mypay)

            # 목표 금액이 입력되지 않은 경우
            if goal_amount is None:
                if request.method == 'POST':
                    # 목표 금액 입력 후 저장
                    goal_amount = int(request.POST['goal_amount'])
                    user.goal_amount = goal_amount  # DB에 저장할 목표 금액 설정
                    user.save()  # 변경사항 DB에 저장
                    return redirect('mypage')  # 저장 후 페이지 새로 고침
            else:
                comparison = "목표 금액 이내로 사용 중입니다." if total_spent <= goal_amount else "목표 금액을 초과했습니다."

            category_percentages = {
                category: round((amount / total_spent) * 100, 2)
                for category, amount in category_totals.items()
            }
            sorted_category_percentages = dict(sorted(category_percentages.items(), key=lambda x: x[1], reverse=True))
            # 90일 이내 만기일 필터링
            expiring_accounts = [
                {
                    "AccountID": account['AccountID'],
                    "balance": account['balance'],
                    "pname" : account['pname'],
                    "ds_rate": float(account['ds_rate']), 
                    "end_date": account['end_date'].strftime("%Y-%m-%d") ,  # 날짜를 JSON 직렬화 가능하도록 문자열로 변환
                    "days_remaining": (account['end_date'] - today).days
                }
                for account in accounts
                if (account['end_date'] - today).days <= 90
            ]
            accounts_list = [
                {
                    "AccountID": account['AccountID'],
                    "balance": account['balance'],
                    "pname": account['pname'],
                    "ds_rate": float(account['ds_rate']),
                    "dstype" : account['dstype'],
                    "end_date": account['end_date'].strftime("%Y-%m-%d"),
                    "days_remaining": (account['end_date'] - today).days
                }
                for account in accounts
            ]
            d_list = [account for account in accounts_list if account['dstype'] == 'd']
            s_list = [account for account in accounts_list if account['dstype'] == 's']
            expiring_accounts_json = json.dumps(expiring_accounts)
        except UserProfile.DoesNotExist:
            pass  # 사용자가 없을 경우 기본값 유지

    # JSON 직렬화
    
    context = {
        'user_name': user_name,          # 사용자 이름
        'accounts': accounts,           # 전체 계좌 정보
        'expiring_accounts': expiring_accounts,  # 만기일 90일 이내 계좌 (리스트 형태)
        'expiring_accounts_json': expiring_accounts_json,  # 만기일 90일 이내 계좌 (JSON 문자열 형태)
        'accounts_list': accounts_list,
        'd_list' : d_list,
        's_list' : s_list,
        'mypay' : mypay,
        'total_spent' : total_spent,
        'category_percentages':json.dumps(sorted_category_percentages),
        'goal_amount' : goal_amount,
        'comparison' : comparison,
    }
    return render(request, 'mypage.html', context)

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

# 함수로 데이터 키 변환 정의
def apply_mapping(data_dict, mapping):
    return {mapping.get(k, k): v for k, v in data_dict.items()}

# 숫자와 %가 있는 부분 추출 함수 (배달앱 포함 일반화)
def extract_percentage_sentences(data, keywords):
    result = []
    for sentence in data:
        # 키워드가 문장에 포함되어 있는지 확인 (앞뒤로 붙는 단어 포함)
        if any(re.search(rf'\b{keyword}\b', sentence) for keyword in keywords):
            # 숫자% 추출
            percentages = re.findall(r'\d+%', sentence)
            if percentages:
                result.extend(percentages)
    return result

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

@login_required_session
def spending_mbti(request):
    customer_id = request.session.get('user_id')  
    user_name = "사용자"
    if customer_id:
        try:
            # CustomerID로 UserProfile 조회
            user = UserProfile.objects.get(CustomerID=customer_id)
            user_name = user.username  # 사용자 이름 설정

            # 'period' 쿼리 파라미터 가져오기
            period = request.GET.get('period', None)


            # 현재 날짜 가져오기
            today = date.today()
            

            # 기간에 따라 시작 날짜 계산
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

            # start_date에서 지난달
            start_date = start_date - relativedelta(months=1)

            # `SpendAmount`에서 기간에 맞는 데이터 필터링
            spend_amounts = SpendAmount.objects.filter(
                CustomerID=customer_id, 
                SDate__gte=start_date  # 시작 날짜 이후의 데이터만 가져옴
            )
            
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
 

            # 여기서 부터는 spendfreq 시작
            # # `SpendFreq`에서 기간에 맞는 데이터 필터링
            spend_freq = SpendFreq.objects.filter(
                CustomerID=customer_id, 
                SDate__gte=start_date  # 시작 날짜 이후의 데이터만 가져옴
            )     

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

            # 소비 예측 모델 불러오기
            mydata_pay = MyDataPay.objects.filter(
                CustomerID=customer_id
            ).values()     
            pd.options.display.float_format = '{:,.2f}'.format
            
            # series 타입을 직렬화
            prediction= senter(mydata_pay)
            # JSON 형식으로 변환
            prediction_dict = prediction.to_dict()
            # Sorting the dictionary by values in descending order
            sorted_prediction = dict(sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True))
            next_month_prediction_json = json.dumps(prediction_dict)

            # 키 변경 매핑 정의
            key_mapping = {
                'allowance': '경조사비',
                'eat': '식비',
                'fashion': '패션/잡화',
                'hobby': '여가/취미',
                'home': '주거비',
                'medical': '의료비',
                'party': '모임회비',
                'phone': '통신비',
                'study': '교육비',
                'transfer': '교통비',
                'predicted_total': '총합'
            }

            # 키 변경을 적용한 새 딕셔너리 생성
            new_prediction_dict = {key_mapping[k]: v for k, v in sorted_prediction.items()}



            # 소비 예측 차트를 위한 값 불러오기
            
            # 직전 1달
            # 현재 월에서 한 달을 빼고 그 월의 첫째 날 계산
            if today.month == 3:
                fred_start_date = today.replace(year=today.year - 3, month=12)
            else:
                fred_start_date = today.replace(month=today.month - 3)

            # start_date에서 지난달
            fred_start_date = fred_start_date - relativedelta(months=1)

            fred_spend_amounts = SpendAmount.objects.filter(
                CustomerID=customer_id ,
                SDate__gte=fred_start_date  # 시작 날짜 이후의 데이터만 가져옴
            ).values()

            # QuerySet에서 리스트로 변환
            fred_spend_amounts_list = list(fred_spend_amounts)

            # 월별 데이터 딕셔너리 생성
            fred_spend_amounts_by_month = {
                item['SDate']: {key: value for key, value in item.items() if key not in ['CustomerID', 'SDate']}
                for item in fred_spend_amounts_list
            }

            # 월별 데이터 쪼개기
            months = list(fred_spend_amounts_by_month.keys())  # 월 목록 생성 (예: ['2024-09', '2024-10', '2024-11'])

            split_month_dict = [
                fred_spend_amounts_by_month[month] for month in months
            ]

            # json1, json2, json3으로 저장
            month1_dict, month2_dict, month3_dict = split_month_dict



            # 키 매핑 정의
            key_mapping = {
                "allowance": "allowance_amount",
                "eat": "eat_amount",
                "fashion": "fashion_amount",
                "hobby": "hobby_amount",
                "home": "home_amount",
                "medical": "medical_amount",
                "party": "party_amount",
                "phone": "phone_amount",
                "study": "study_amount",
                "transfer": "transfer_amount",
                "predicted_total": "TotalAmount"
            }
            # key_mapping을 반대로 변환
            reversed_key_mapping = {value: key for key, value in key_mapping.items()}
            
            # 반전된 key_mapping을 사용하여 각 월별 데이터 변환
            month1_dict_map = apply_mapping(month1_dict, reversed_key_mapping)
            month2_dict_map = apply_mapping(month2_dict, reversed_key_mapping)
            month3_dict_map = apply_mapping(month3_dict, reversed_key_mapping)

            # JSON으로 변환
            month1_json = json.dumps(month1_dict_map)
            month2_json = json.dumps(month2_dict_map)
            month3_json = json.dumps(month3_dict_map)


            # 월 추출
            result = []
            prev_year = None
            # 마지막 달 계산 및 다음 달 추가
            last_month = months[-1]
            year, month = map(int, last_month.split('-'))

            if month == 12:  # 마지막 달이 12월인 경우
                next_year = year + 1
                next_month = 1
            else:  # 12월이 아닌 경우
                next_year = year
                next_month = month + 1

            # 다음 달 추가
            next_month_str = f"{next_year}-{str(next_month).zfill(2)}"
            months.append(next_month_str)
            # 리스트를 JSON으로 변환
            months_json = json.dumps(months)

            # print('months',months)
            
            # 추가적으로 월 형식으로 바꾸고 싶을 때 사용할 코드
            # for date in months:
            #     year, month = date.split('-')
            #     if prev_year != year:  # 해가 바뀌는 경우
            #         korean_month = f"{str(year)[-2:]}년 {int(month)}월 "  # 한국어 형식: 1월 (25년)
            #     else:  # 같은 해인 경우
            #         korean_month = f"{int(month)}월"  # 한국어 형식: 1월
            #     result.append(korean_month)
            #     prev_year = year

            # # 결과 출력
            # print("Processed months:", result)

            

            # 카드 추천 만들기
            # 대부분이 식비이긴한데, 일단 소비 위주로 맞추는게 좋지 않을까?

            # 결과 확인
            top_card_list = [key for key, _ in sorted(new_prediction_dict.items(), key=lambda x: x[1], reverse=True)[1:4]]


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
                AmountNum = round(AmountNum / 10) * 10
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
                discount = round(AmountNum * max_value * 0.01 / 10) * 10

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

        except UserProfile.DoesNotExist:
            pass  # 사용자가 없을 경우 기본값 유지

    next_month_prediction_json = json.dumps(next_month_prediction_json, ensure_ascii = False)
    context = {
        'user_name': user_name,
        'sorted_categories_json' : sorted_categories_json, 
        'amount_total_json' : amount_total_json, 
        'Freq_sorted_categories_json' : Freq_sorted_categories_json,
        'Freq_total_json' : Freq_total_json,
        'next_month_prediction_json' : next_month_prediction_json,
        'month1_json' : month1_json,
        'month2_json' : month2_json,
        'month3_json' : month3_json,
        'months_json' : months_json,
        'card_results_json' : card_results_json,
        'card_detail_results': card_detail_results,
        'top_card_list': top_card_list,
    }

    return render(request, 'spending_mbti.html', context)

def main(request):
    today = timezone.now().date()
    yesterday = today - timedelta(days=1)

    # 어제 날짜의 이미지 데이터 가져오기
    wc_entry = Wc.objects.filter(date=yesterday).first()
    image_base64 = base64.b64encode(wc_entry.image).decode('utf-8') if wc_entry else None

    # 어제 날짜의 뉴스 데이터 가져오기
    news_entries_queryset = News.objects.filter(
        ndate=yesterday, 
        summary__isnull=False
    )

    # 중복 제거 로직
    seen_titles = set()  # 이미 본 제목을 저장
    news_entries = []
    for news in news_entries_queryset:
        if news.title not in seen_titles:
            seen_titles.add(news.title)
            news_entries.append({'title': news.title, 'summary': news.summary, 'url': news.url })

    # 디버깅용 출력

    context = {
        'image_base64': image_base64,
        'news_entries': news_entries,
    }

    return render(request, 'main.html', context)

@login_required_session
def report_ex(request):
    customer_id = request.session.get('user_id')  
    user_name = "사용자"  # 기본값 설정

    if customer_id:
        try:
            # CustomerID로 UserProfile 조회
            user = UserProfile.objects.get(CustomerID=customer_id)
            user_name = user.username  # 사용자 이름 설정
        except UserProfile.DoesNotExist:
            pass  # 사용자가 없을 경우 기본값 유지

    context = {
        'user_name': user_name,
    }
    return render(request, 'report_ex.html', context)

def assign_cluster(stage_class, sex, age):
    if stage_class == 0:
        if sex == 'M' and age in [19, 20, 21]:
            return [5, 2, 1, 4]
        else:
            return [0, 1, 4]
    else:
        return [1, 4]

def get_top_data_by_customer_class(stageclass, inlevel):
    # Elasticsearch 연결
    
    
    stageclass = stageclass
    inlevel = inlevel
    headers = {
        "Content-Type": "application/json"
    }
    # Elasticsearch 쿼리
    query = {
        "query": {
            "bool": {
                "filter": [
                    {"term": {"customer_class.Stageclass.keyword": stageclass}},  # keyword로 정확 매칭
                    {"term": {"customer_class.Inlevel": inlevel}}
                ]
            }
        },
        "aggs": {
            "group_by_data": {
                "terms": {
                    "script": {
                    "source": """
                        def data = doc['data.product_name.keyword'].value + '|' +
                                doc['data.bank.keyword'].value + '|' +
                                doc['data.baser.keyword'].value + '|' +
                                doc['data.maxir.keyword'].value + '|' +
                                doc['data.method.keyword'].value;
                        return data;
                    """,
                    "lang": "painless"
                    },
                    "size": 3,
                    "order": {"_count": "desc"}
                },
                "aggs": {
                    "top_hits": {
                        "top_hits": {
                            "size": 1,
                            "_source": {
                            "includes": ["data", "customer_class", "timestamp"]
                            }
                        }
                    }
                }
            }
        },
        "size": 0
    }

    try:
        # Elasticsearch에서 데이터 가져오기
        response = es.search(index="ps_product_click_logs", body=query)
        # 집계 데이터 추출
        aggs_results = response.get("aggregations", {}).get("group_by_data", {}).get("buckets", [])

        # 상위 3개 데이터만 추출
        top_data = []
        for bucket in aggs_results:
            top_hit = bucket.get("top_hits", {}).get("hits", {}).get("hits", [])
            if top_hit:
                top_data.append({
                    "data": top_hit[0]["_source"]["data"],
                    "count": bucket["doc_count"]  # 해당 데이터의 카운트
                })

        return top_data

    except Exception as e:
        # 오류 처리
        return JsonResponse({"error": str(e)}, status=500)





@login_required_session
def summary_view(request):
    customer_id = request.session.get('user_id')  # 세션에서 CustomerID 가져오기
    today = timezone.now().date()
    yesterday = today - timezone.timedelta(days=1)

    # 워드클라우드 이미지 가져오기
    wc_entry = Wc.objects.filter(date=yesterday).first()
    image_base64 = base64.b64encode(wc_entry.image).decode('utf-8') if wc_entry else None

    # 어제 날짜의 뉴스 데이터 가져오기
    news_entries_queryset = News.objects.filter(ndate=yesterday, summary__isnull=False)
    seen_titles = set()
    news_entries = []
    for news in news_entries_queryset:
        if news.title not in seen_titles:
            seen_titles.add(news.title)
            news_entries.append({
                'title': news.title,
                'summary': news.summary,
                'url': news.url
            })

    # 사용자 정보 가져오기
    user_name = "사용자"  # 기본값 설정
    if customer_id:
        try:
            user = UserProfile.objects.get(CustomerID=customer_id)
            user_name = user.username
        except UserProfile.DoesNotExist:
            pass

    if not customer_id:  # 로그인되지 않은 사용자는 로그인 페이지로 리디렉션
        return redirect('login')
    
    # Average 테이블에서 고객 소득분위 기준 데이터 조회
    average_data = Average.objects.filter(
        stageclass=user.Stageclass,
        inlevel=user.Inlevel
    ).first()

    user_asset_data = MyDataAsset.objects.filter(CustomerID=customer_id).first()

    average_values = {
    '총자산': (average_data.asset + average_data.finance),
    '현금자산': average_data.finance,
    '수입': average_data.income,
    '지출': average_data.spend
}
    user_data = {
    '총자산': user_asset_data.total,
    '현금자산': user_asset_data.financial,
    '수입': user_asset_data.monthly_income,
    '지출': user_asset_data.expenses
}
    # 추천 상품 처리
    recommended_products = Recommend.objects.filter(CustomerID=customer_id)
    recommended_count = recommended_products.count()
    recommended_dsid_list = {'dproduct': [], 'sproduct': []}

    if recommended_count > 0:
        # DProduct와 SProduct 각각 추천 가져오기
        recommended_dsid_list['dproduct'] = list(recommended_products.filter(dproduct__isnull=False).values_list('dproduct', flat=True))
        recommended_dsid_list['sproduct'] = list(recommended_products.filter(sproduct__isnull=False).values_list('sproduct', flat=True))

    # 추천 상품 세부 정보 가져오기
    recommended_product_details = list(
        DProduct.objects.filter(dsid__in=recommended_dsid_list['dproduct']).values('dsid','dsname', 'bank', 'baser', 'maxir')
    ) + [
        {   
            'dsid' : sp['dsid'],
            'dsname': sp['product_name'],
            'bank': sp['bank_name'],
            'baser': sp['base_rate'],
            'maxir': sp['max_preferential_rate']
        }
        for sp in SProduct.objects.filter(DSID__in=recommended_dsid_list['sproduct']).values('DSID','product_name', 'bank_name', 'base_rate', 'max_preferential_rate')
    ]


    # 랜덤 상품 추가
    if recommended_count < 5:
        remaining_count = 5 - recommended_count
        random_dproducts = DProduct.objects.exclude(dsid__in=recommended_dsid_list['dproduct']).order_by('?')[:remaining_count]
        random_sproducts = SProduct.objects.exclude(DSID__in=recommended_dsid_list['sproduct']).order_by('?')[:remaining_count]

        random_product_details = list(random_dproducts.values('dsid','dsname', 'bank', 'baser', 'maxir')) + [
            {
                'dsid' : sp.DSID,
                'dsname': sp.product_name,
                'bank': sp.bank_name,
                'baser': sp.base_rate,
                'maxir': sp.max_preferential_rate
            }
            for sp in random_sproducts
        ]

        product_details = recommended_product_details + random_product_details
    else:
        product_details = recommended_product_details

    # 중복 제거 및 최대 5개 제한
    unique_product_details = {p['dsname']: p for p in product_details if p['dsname']}.values()
    product_details = list(unique_product_details)[:5]


    ## 적금 추천 상품 top 3
    # Django ORM을 사용하여 데이터 가져오기
    cluster_savings = SProduct.objects.all()

    # 필요한 데이터를 Pandas DataFrame으로 변환
    

    data = list(cluster_savings.values())  # ORM QuerySet을 리스트로 변환
    cluster_savings = pd.DataFrame(data)
    # 결과를 저장할 빈 데이터프레임 생성 (모든 열 포함)
    final_result = pd.DataFrame(columns=cluster_savings.columns)


    birth_year = user.Birth.year  # 주민번호 앞자리로 연도 추출
    current_year = datetime.now().year
    age = current_year - birth_year
    cluster = assign_cluster(user.Stageclass, user.sex, age)

    for i in cluster:
        filtered_df = cluster_savings[cluster_savings['cluster1'] == i]
        if not filtered_df.empty:
            sorted_df = filtered_df.sort_values(by=['max_preferential_rate', 'base_rate'], ascending=[False, False])
            if not sorted_df.empty:
                top_result = sorted_df.head(5)
                final_result = pd.concat([final_result, top_result], ignore_index=True)

    # 적금 최종 추천 3개로 제한
    final_recommend_json = final_result.head(5)[["DSID","product_name", "bank_name", "max_preferential_rate", "base_rate", "signup_method"]].to_dict(orient='records')
    

    # 예금 추천 처리
    cluster_scores = {i: 0 for i in range(7)}
    if user.Stageclass in [0, 1, 2, 3]:
        for cluster in [2, 4, 5, 6]:
            cluster_scores[cluster] += 1
    elif user.Stageclass in [4, 5, 6, 7]:
        for cluster in [0, 1, 2, 3, 4, 5, 6]:
            cluster_scores[cluster] += 1
    if user.Inlevel in [0, 1]:
        for cluster in [0, 1, 2, 6]:
            cluster_scores[cluster] += 1
    elif user.Inlevel in [2, 3, 4]:
        for cluster in [3, 4, 5]:
            cluster_scores[cluster] += 1

    sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    top_clusters = [cluster[0] for cluster in sorted_clusters]
    filtered_results = []

    for cluster in top_clusters:
        filtered_deposits_query = DProduct.objects.filter(cluster=cluster).values('dsid', 'name', 'bank', 'baser', 'maxir','method')
        filtered_results.append(pd.DataFrame(filtered_deposits_query))
    request.session['clusters'] = top_clusters
    final_recommendations = pd.concat(filtered_results, ignore_index=True)
    # 중복 제거
    final_recommendations_drop_duplicates = final_recommendations.drop_duplicates(subset=["name", "bank", "baser", "maxir", "method"])
    top2 = final_recommendations_drop_duplicates.sort_values(by='maxir', ascending=False).head(5)
    deposit_recommend_dict = top2.to_dict(orient='records')
    final_recommend_with_logo = [
        {**item, "logo": get_bank_logo(item.get("bank_name", ""))} for item in final_recommend_json
    ]
    deposit_recommend_with_logo = [
        {**item, "logo": get_bank_logo(item.get("bank", ""))} for item in deposit_recommend_dict
    ]
    request.session['final_recommend'] = final_recommend_with_logo[:5]  # 적금 Top 5
    request.session['deposit_recommend'] = deposit_recommend_with_logo[:5]  # 예금 Top 5

    final_recommend_display = final_recommend_json[:2]  # 적금 2개
    deposit_recommend_display = deposit_recommend_dict[:3]  # 예금 3개
    
    
    # 로그 데이터 확인 
    log_cluster = get_top_data_by_customer_class(user.Stageclass, user.Inlevel)

    # "data" 부분만 추출
    filtered_data = [item['data'] for item in log_cluster]

    # JSON으로 변환
    filtered_data_json = json.dumps(filtered_data, ensure_ascii=False)


    # 최종 데이터 전달
    context = {
        'product_details': product_details,
        'image_base64': image_base64,
        'news_entries': news_entries,
        'user_name': user_name,
        'final_recommend': final_recommend_display,  # 적금 Top 3
        'deposit_recommend': deposit_recommend_display,  # 예금 Top 2
        'average_data': json.dumps(average_values, ensure_ascii=False),
        'user_data': json.dumps(user_data, ensure_ascii=False),
        
    }

    return render(request, 'loginmain.html', context)

@login_required_session
def info(request):
    customer_id = request.session.get('user_id')
    user_name = "사용자"

    if customer_id:
        try:
            user = UserProfile.objects.get(CustomerID=customer_id)
            user_name = user.username
        except UserProfile.DoesNotExist:
            pass

    context = {'user_name': user_name}

    if request.method == 'POST':
        saving_method = request.POST.get('saving_method')
        bank_option = request.POST.get('bank_option')
        selected_preferences = request.POST.getlist('preferences')
        period = request.POST.get('period')
        cluster_list = request.session.get('clusters', [])

        if not cluster_list:
            return render(request, 'error.html', {'message': 'Cluster 값이 없습니다.'})


        # 은행 유형 필터링
        s_bank_query = Q()
        if bank_option == "일반은행":
            s_bank_query = Q(bank_name__icontains="1금융권")
        elif bank_option == "일반은행 + 저축은행":
            pass
        elif bank_option:
            return render(request, 'error.html', {'message': '유효하지 않은 은행 옵션입니다.'})
        
        d_bank_query = Q()
        if bank_option == "일반은행":
            d_bank_query = ~Q(bank__icontains="저축은행")
        elif bank_option == "일반은행 + 저축은행":
            pass
        elif bank_option:
            return render(request, 'error.html', {'message': '유효하지 않은 은행 옵션입니다.'})
        

        # 클러스터 필터링
        d_cluster_query = Q()
        s_cluster_query = Q()
        for cluster in cluster_list:
            d_cluster_query |= Q(cluster=cluster)
            s_cluster_query |= Q(cluster1=cluster)


        d_period_query = Q()
        s_period_query = Q()
        # 기간 필터링
        if period in ["12", "24", "36"]:
            d_period_query = (
                Q(mindate__lte=int(period)) & Q(maxdate__gte=int(period)) &
                ~Q(mindate=F('maxdate'))  # mindate와 maxdate가 같은 경우 제외
            ) | Q(mindate=F('maxdate'))  # mindate가 maxdate와 같다면 선택
            s_period_query =  (
                Q(min_period=int(period)) & Q(max_period__gte=int(period)) &
                ~Q(min_period=F('max_period')) 
            ) | Q(min_period=F('max_period')) 
        else:
            pass
        

        # 우대조건 필터링
        d_preference_query = Q(condit__icontains="해당없음")
        s_preference_query = Q(preferential_conditions__icontains="해당없음")

        for preference in selected_preferences:
            d_preference_query |= Q(condit__icontains=preference)  # LIKE '%preference%'
            s_preference_query |= Q(preferential_conditions__icontains=preference)  # LIKE '%preference%'
        # 적립 방법에 따라 추천 결과 생성
        deposit_recommend = []
        final_recommend = []

        if saving_method == "목돈 모으기":
            deposit_recommend = DProduct.objects.filter(d_bank_query & d_preference_query & d_cluster_query & d_period_query).order_by('-maxir', '-name').values()[:5]
            deposit_recommend = add_bank_logo(deposit_recommend, 'bank')
                        # 중복 제거
            ## 중복 제거 키 설정
            keys_to_check = ["name", "bank", "baser", "maxir"]

            ## 데이터 중복 제거
            # 데이터프레임으로 변환
            df = pd.DataFrame(deposit_recommend)
            # 중복 제거
            df = df.drop_duplicates(subset=keys_to_check)
            # 다시 딕셔너리 리스트로 변환
            deposit_recommend = df.to_dict(orient='records')


        elif saving_method == "목돈 굴리기":
            final_recommend = SProduct.objects.filter(s_bank_query & s_preference_query & s_cluster_query & s_period_query).order_by('-max_preferential_rate', '-bank_name').values()[:5]
            final_recommend = add_bank_logo(final_recommend, 'bank_name')


            
        elif saving_method == "목돈 모으기 + 목돈 굴리기":
            deposit_recommend = DProduct.objects.filter(d_preference_query & d_bank_query & d_cluster_query & d_period_query).order_by('-maxir', '-name').values()[:5]
            deposit_recommend = add_bank_logo(deposit_recommend, 'bank')
            final_recommend = SProduct.objects.filter(s_bank_query & s_preference_query & s_cluster_query & s_period_query).order_by('-max_preferential_rate', '-base_rate','-bank_name').values()[:5]
            final_recommend = add_bank_logo(final_recommend, 'bank_name')
                                    # 중복 제거
            ## 중복 제거 키 설정
            keys_to_check = ["name", "bank", "baser", "maxir"]

            ## 데이터 중복 제거
            # 데이터프레임으로 변환
            df = pd.DataFrame(deposit_recommend)
            # 중복 제거
            df = df.drop_duplicates(subset=keys_to_check)
            # 다시 딕셔너리 리스트로 변환
            deposit_recommend = df.to_dict(orient='records')

        else:
            return render(request, 'error.html', {'message': '유효하지 않은 적립 방법입니다.'})
        # 추천 결과를 세션에 JSON 형식으로 저장
        if deposit_recommend:
            request.session['deposit_recommend'] = json.dumps(list(deposit_recommend), cls=DjangoJSONEncoder)
        if final_recommend:
            request.session['final_recommend'] = json.dumps(list(final_recommend), cls=DjangoJSONEncoder)

        # 목표 금액

        # 가입 기간
        
        # 우대 조건 적용

        context.update({
            'deposit_recommend': deposit_recommend,
            'final_recommend': final_recommend,
            # 'select_deposit_recommend' : select_deposit_recommend,
            # 'select_final_recommend' :select_final_recommend,
        })
        return redirect('top5')

    # GET 요청일 경우 템플릿 렌더링
    return render(request, 'savings_info.html', context)


@login_required_session
def top5(request):
    customer_id = request.session.get('user_id')  
    user_name = "사용자"  # 기본값 설정

    if customer_id:
        try:
            # CustomerID로 UserProfile 조회
            user = UserProfile.objects.get(CustomerID=customer_id)
            user_name = user.username  # 사용자 이름 설정
        except UserProfile.DoesNotExist:
            pass  # 사용자가 없을 경우 기본값 유지

    # 세션에서 추천 데이터를 가져오기
    final_recommend = request.session.get('final_recommend', '[]')
    deposit_recommend = request.session.get('deposit_recommend', '[]')

    # JSON 역직렬화 (문자열일 경우)
    try:
        final_recommend = json.loads(final_recommend) if isinstance(final_recommend, str) else final_recommend
        deposit_recommend = json.loads(deposit_recommend) if isinstance(deposit_recommend, str) else deposit_recommend
    except json.JSONDecodeError:
        final_recommend = []
        deposit_recommend = []

    # 로그 데이터 확인 
    log_cluster = get_top_data_by_customer_class(user.Stageclass, user.Inlevel)
    # "data" 부분만 추출
    filtered_data = list([item['data'] for item in log_cluster])
    # 은행 이름에 해당하는 로고 파일명을 매핑
    filtered_data_with_logo = [
        {**item, "logo": get_bank_logo(item.get("bank", ""))} for item in filtered_data
    ]


    # # JSON으로 변환
    # # filtered_data_json = json.dumps(filtered_data, ensure_ascii=False)
    # try:
    #     filtered_data = json.loads(filtered_data) if isinstance(filtered_data, str) else filtered_data
    
    # except json.JSONDecodeError:
    #     filtered_data = []
    # print('filtered_data',filtered_data)

    # request.session['filtered_data'] = filtered_data

    # filtered_data_json = json.dumps(filtered_data, ensure_ascii=False)


    # Context에 데이터 추가
    context = {
        'user_name': user_name,
        'final_recommend': final_recommend,  # 적금 Top 3 (로고 포함)
        'deposit_recommend': deposit_recommend,  # 예금 Top 2 (로고 포함)
        'filtered_data' : filtered_data_with_logo,
    }

    return render(request, 'recommend_savings_top5.html', context)

@login_required_session
def main_view(request):
    if request.user.is_authenticated:
        try:
            # usertable에서 username을 기준으로 조회하여 CustomerID 가져오기
            user = UserProfile.objects.get(username=request.user.username)
            user_id = user.CustomerID  # MySQL 데이터베이스의 CustomerID 필드를 user_id로 사용
        except UserProfile.DoesNotExist:
            user_id = "anonymous"  # 사용자가 없을 경우 기본값 설정
    else:
        user_id = "anonymous"

    session_id = request.session.session_key

    # 메인 페이지 접근 로그 기록
    log_user_action(user_id=user_id, session_id=session_id, action="page_view", page="main")

    return render(request, 'main.html')

@csrf_exempt
def log_click_event(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        event_data = {
            "event": data.get("event"),
            "timestamp": data.get("timestamp")
        }
        # Elasticsearch에 로그 저장
        es.index(index="django_logs", body=event_data)
        return JsonResponse({"status": "success"})
    return JsonResponse({"status": "failed"}, status=400)

@login_required_session
def originreport_page(request):
    # 세션에서 사용자 ID 가져오기
    customer_id = request.session.get('user_id')
    user_name = "사용자"  # 기본값 설정

    if customer_id:
        try:
            # CustomerID로 UserProfile 조회
            user = UserProfile.objects.get(CustomerID=customer_id)
            user_name = user.username  # 사용자 이름 설정
        except UserProfile.DoesNotExist:
            pass  # 사용자가 없을 경우 기본값 유지
    try:
        # 세션 데이터가 없는 경우 예외 발생
        if not customer_id:
            raise ValueError("로그인이 필요합니다.")

        # CustomerID로 UserProfile 조회
        user = UserProfile.objects.get(CustomerID=customer_id)
        cnow = datetime.now()
        # 한 달 전 날짜 계산
        last_month = cnow - timedelta(days=30)  # 30일을 빼서 대략적으로 한 달을 계산
        last_month_str = last_month.strftime("%Y-%m")  # 형식에 맞게 문자열로 변환

        # Average 테이블에서 고객 소득분위 기준 데이터 조회
        average_data = Average.objects.filter(
            stageclass=user.Stageclass,
            inlevel=user.Inlevel
        ).first()
        user_asset_data = MyDataAsset.objects.filter(CustomerID=customer_id).first()
        spend_freq = SpendFreq.objects.filter(CustomerID=customer_id, SDate=(last_month_str)).first()
        spend_amount = SpendAmount.objects.filter(CustomerID=customer_id, SDate=(last_month_str)).first()

        if not average_data:
            raise ValueError(f"소득 분위 데이터가 없습니다. (Stage Class: {user.Stageclass}, Inlevel: {user.Inlevel})")

        # MyData에서 고객 데이터 조회
        if not user_asset_data:
            raise ValueError(f"사용자 데이터를 찾을 수 없습니다. (Customer ID: {customer_id})")
        if not spend_amount:
            raise ValueError(f"사용자 데이터를 찾을 수 없습니다. (Customer ID: {customer_id})")
        if not spend_freq:
            raise ValueError(f"사용자 데이터를 찾을 수 없습니다. (Customer ID: {customer_id})")

         # 분석 로직
        # 개인 자산 데이터 기반 계산
        net_income = user_asset_data.total - user_asset_data.debt  # 순자산
        financial_ratio = user_asset_data.financial / user_asset_data.total * 100  # 금융자산 비율(%)
        real_estate_ratio = user_asset_data.estate / user_asset_data.total * 100  # 부동산 자산 비율(%)
        other_asset_ratio = user_asset_data.ect / user_asset_data.total * 100  # 기타 자산 비율(%)
        NLAR = (user_asset_data.estate + user_asset_data.ect) / user_asset_data.total * 100  # 비유동자산 비율(%)

        # 평균그룹 순자산
        group_net_income = average_data.asset - average_data.debt

        # 그룹별 자산유형별 비중
        group_financial_ratio = average_data.finance / average_data.asset * 100  # 금융자산 비율(%)
        group_real_estate_ratio = average_data.estate / average_data.asset * 100  # 부동산 자산 비율(%)
        group_other_asset_ratio = average_data.etc / average_data.asset * 100  # 기타 자산 비율(%)

        # 부채 비율 및 유동성 비율
        debt_ratio = user_asset_data.debt / user_asset_data.total * 100  # 부채 비율(%)
        LR = user_asset_data.financial / user_asset_data.debt  # 유동성비율

        # 저축률 계산
        SR = user_asset_data.saving / user_asset_data.monthly_income * 100  # 월 저축률(%)
        TSR = user_asset_data.financial / user_asset_data.total * 100  # 총 저축률(%)

        # 소득대비 자산 비율
        IAR = user_asset_data.total / user_asset_data.total_income  # 소득 대비 자산 비율

        # 그룹과의 비교
        income_difference = user_asset_data.monthly_income - average_data.income  # 소득 차이
        spend_difference = (spend_amount.TotalAmount + user_asset_data.rent) - average_data.spend  # 지출 차이
        financial_comparison = user_asset_data.financial - average_data.finance  # 금융자산 차이

        # 제외할 키를 명시적으로 정의
        excluded_keys = {'CustomerID', 'SDate', 'TotalAmount'}

        # 함수 호출하여 JSON 데이터 생성
        period = '1m'


            # 현재 날짜 가져오기
        today = date.today()
        

        # 기간에 따라 시작 날짜 계산
        if period == '1m':
            # 직전 1달
            # 현재 월에서 한 달을 빼고 그 월의 첫째 날 계산
            if today.month == 1:
                start_date = today.replace(year=today.year - 1, month=12)
            else:
                start_date = today.replace(month=today.month - 1)


        # start_date에서 지난달
        start_date = start_date - relativedelta(months=1)
        asset_data = MyDataAsset.objects.filter(CustomerID=customer_id).values('estate', 'financial', 'ect')
        mydata_assets_list = list(asset_data)  # QuerySet을 리스트로 변환
        mapped_assets = {
            "부동산": mydata_assets_list[0]["estate"] if mydata_assets_list else 0,
            "금융": mydata_assets_list[0]["financial"] if mydata_assets_list else 0,
            "기타": mydata_assets_list[0]["ect"] if mydata_assets_list else 0,
        }

        # JSON으로 변환
        mydata_assets_json = json.dumps(mapped_assets, ensure_ascii=False)
        # `SpendAmount`에서 기간에 맞는 데이터 필터링
        spend_amounts = SpendAmount.objects.filter(
            CustomerID=customer_id, 
            SDate__gte=start_date  # 시작 날짜 이후의 데이터만 가져옴
        )
        
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

        # sorted_categories와 amount_total을 JSON으로 변환
        sorted_categories_json = json.dumps(sorted_categories)
        # TotalAmount 값 검증 및 정수로 변환
        total_amount = int(spend_amount.TotalAmount if spend_amount.TotalAmount else 0)

        if total_amount == 0:
            raise ValueError("spend_amount['TotalAmount'] 값이 0이거나 없습니다.")

        # 카테고리별 소비 비중 계산
        # spend_amount 객체의 속성들 중 excluded_keys에 포함되지 않은 것만 처리
        category_spend_ratios = {}
        for category in spend_amount.__dict__:  # __dict__를 사용하여 객체의 실제 속성에 접근
            if category not in excluded_keys:
                value = getattr(spend_amount, category, 0)  # 속성 값 가져오기
                
                # 숫자 값에 대해서만 비율 계산
                if isinstance(value, (int, float)):
                    category_spend_ratios[category] = (value / total_amount * 100) if total_amount != 0 else 0
                else:
                    category_spend_ratios[category] = 0

        # 그룹 소비 비중
        group_category_ratios = {
            category: getattr(average_data, category, 0) if isinstance(getattr(average_data, category, 0), (int, float)) else 0
            for category in category_spend_ratios
        }

        # 소비 차이 분석
        excessive_categories = {
            category: category_spend_ratios[category] - group_category_ratios.get(category, 0)
            for category in category_spend_ratios
            if isinstance(category_spend_ratios[category], (int, float)) and category_spend_ratios[category] > group_category_ratios.get(category, 0)
        }

        # 개인 및 그룹 소득대비 지출 비율
        personal_spend_ratio = (spend_amount.TotalAmount + user_asset_data.rent) / user_asset_data.monthly_income  # 개인 지출 비율
        group_spend_ratio = average_data.spend / average_data.income  # 그룹 지출 비율


        # 데이터 준비
        bar_data = {
            '총자산': user_asset_data.total,
            '현금자산': user_asset_data.financial,
            '수입': user_asset_data.monthly_income,
            '지출': abs(spend_amount.TotalAmount)
        }
        average_values = {
            '총자산': (average_data.asset + average_data.finance),
            '현금자산': average_data.finance,
            '수입': average_data.income,
            '지출': average_data.spend
        }

        # 데이터 정리
        analysis_results = {
            "net_income" : net_income,
            "financial_ratio": financial_ratio,
            "real_estate_ratio": real_estate_ratio,
            "other_asset_ratio": other_asset_ratio,
            "NLAR": NLAR,
            "group_financial_ratio": group_financial_ratio,
            "group_real_estate_ratio": group_real_estate_ratio,
            "group_other_asset_ratio": group_other_asset_ratio,
            "group_net_income": group_net_income,
            "debt_ratio": debt_ratio,
            "LR": LR,
            "SR": SR,
            "TSR": TSR,
            "IAR": IAR,
            "income_difference": income_difference,
            "spend_difference": spend_difference,
            "financial_comparison": financial_comparison,
            "excessive_categories": excessive_categories,
            "personal_spend_ratio": personal_spend_ratio,
            "group_spend_ratio": group_spend_ratio,
            "category_spend_ratios": category_spend_ratios
        }

        # 프롬프트 생성
        prompt = f"""
        당신은 금융 데이터 분석 전문가인 '만덕이'입니다. 고객의 자산, 소득, 지출 데이터를 기반으로 개인화된 금융 생활 분석 및 개선 리포트를 작성하세요.
        만덕이는 긍정적이고 따뜻하게 응원해주는 오리로, 고객이 스스로를 격려하며 개선 방향을 이해할 수 있도록 설명합니다.
        예를 들어, "오! 정말 잘하고 있어요! 조금만 더 이렇게 하면 완벽할 거예요"처럼 친절하고 귀여운 말투를 사용하세요.
        리포트 작성 시 한 문장 마다 띄어쓰기해주세요. 리포트 각 항목마다 앞의 '-' 표시는 지우고 출력해주세요. 금액을 표시할때는 만단위로 말해주세요.
        총 글자수는 2500자를 넘지 않도록 해주세요. 


        - 자산 정보:
                - 총자산: {user_asset_data.total}
                - 금융자산: {user_asset_data.financial}
                - 부동산자산: {user_asset_data.estate}
                - 기타 자산: {user_asset_data.ect}
                - 부채 : {user_asset_data.debt}
                - 월소득: {user_asset_data.monthly_income}
                - 소비 데이터:
                - 전체 지출 건수: {spend_freq.TotalFreq}
                - 전체 지출 금액: {spend_amount.TotalAmount}
                - 카테고리별 소비 빈도: --debug {spend_freq}
                - 카테고리별 소비 금액: {spend_amount}
                - 그룹 평균 데이터:
                - 수입: {average_data.income}
                - 총자산: {average_data.asset}
                - 금융자산: {average_data.finance}
                - 부채: {average_data.debt}
                - 총지출: {average_data.spend}
                - 소비 카테고리별 비중: {average_data}

        - 자산 및 소비 현황 판단 지표
                    - 순자산: {analysis_results['net_income']}
                    - 금융자산 비율: {analysis_results['financial_ratio']}
                    - 부동산 자산 비율: {analysis_results['real_estate_ratio']}
                    - 기타 자산 비율: {analysis_results['other_asset_ratio']}
                    - 비유동자산 비중: {analysis_results['NLAR']}
                    - 그룹별 금융자산 비율: {analysis_results['group_financial_ratio']}
                    - 그룹별 부동산 자산 비율: {analysis_results['group_real_estate_ratio']}
                    - 그룹별 기타 자산 비율: {analysis_results['group_other_asset_ratio']}
                    - 평균그룹 순자산: {analysis_results['group_net_income']}
                    - 부채비율: {analysis_results['debt_ratio']}
                    - 유동성비율: {analysis_results['LR']}
                    - 월저축률: {analysis_results['SR']}
                    - 총자산저축률: {analysis_results['TSR']}
                    - 소득대비 자산 비율: {analysis_results['IAR']}
                    - 평균과의 소득차이: {analysis_results['income_difference']}
                    - 평균과의 지출차이: {analysis_results['spend_difference']}
                    - 평균과의 금융자산차이: {analysis_results['financial_comparison']}
                    - 카테고리별 소비 비중: {analysis_results['category_spend_ratios']}
                    - 과소비 카테고리: {analysis_results['excessive_categories']}
                    - 개인 소득대비 지출비중: {analysis_results['personal_spend_ratio']}
                    - 그룹 소득대비 지출비중: {analysis_results['group_spend_ratio']}

        ### **리포트 구성 항목**
        안녕하세요-! 당신의 금융 파트너 만덕입니다-!
        제가 당신의 금융 생활을 분석해왔어요-! 우리 같이 살펴볼까요? 🤗

        #### **자산 현황 분석**
        [줄바꿈]
        비유동자산, 유동성비율 등 조금 어려운 금융용어는 쉬운 설명으로 바꿔서 출력해주세요. 
        고객의 자산현황을 분석하기위한 평가 지표는 다음과 같습니다. 
        순자산이 평균 그룹의 순자산보다 많은지 비교하세요. 평균보다 높다면 잘하고 있는겁니다. 금융자산, 부동산자산, 기타자산의 비중을 정리하고 각 자산의 비중을 그룹별 자산비중과 비교합니다. 비유동자산의 비율이 50이하면 좋음, 그렇지 않으면 위험이라는 것을 기준으로 비유동자산 상황을 분석합니다. 부채비율은 40 이하(이상적), 40-80(주의 필요), 80 이상(위험)을 기준으로 부채비율을 평가합니다. 유동성 비율은 2 이상(매우 양호), 1.5-2(양호), 1.0-1.5(주의 필요), 1.0 이하(위험)의 기준으로 평가합니다.  
        고객에게 보여주는 리포트에는 고객의 자산이나 비율만 표시하고 기준점은 따로 설명하지 않습니다. 
        전반적인 평가를 모두 합쳐 고객의 자산상황을 평가하고 응원의 메세지와 함께 개선방향을 제시해주세요. 

        #### **저축 및 소비 습관 분석**
        [줄바꿈]
        월저축률(월소득 대비 저축률)이 10-20% 미만이면 위험, 10-20%는 최소 충족, 50-60%가 이상적임을 기준으로 평가합니다. 50-60% 이하의 월 저축률은 조금 더 개선이 필요한 부분입니다. 월저축률이 최소충족 비율보다 높더라도 이상적인 기준보다 낮다면 저축을 늘릴 필요가 있다는 점을 알려주세요. 총자산저축률은 20-30%면 양호한 수준이지만 저축을 조금 더 늘리도록 장려할 필요가 있습니다. 소득 대비 자산 비율은 3-5가 양호, 5 이상은 아주 좋은 수준, 3 미만은 자산 형성이 필요한 상태임을 기준으로 평가합니다.
        고객에게 보여주는 리포트에는 고객의 자산이나 비율만 표시하고 기준점은 따로 설명하지 않습니다.
        고객의 전체적인 저축 상태를 진단하고, 저축 상태가 보통 또는 보통이하(위험) 수준이면 저축을 장려하는 응원 멘트화 함께 개선방향을 제시해주세요.

        #### **소비 분석**
        [줄바꿈]
        평균과의 소비 비교 : 평균보다 많이 지출하거나 적게 지출하는지를 기준으로 고객의 소비 습관을 판단합니다. 카테고리별 소비 차이 분석 : 소비 카테고리별 비중을 분석해 결제 빈도별, 결제 금액별 가장 소비가 많은 상위 5개 카테고리를 나열하고, 가장 많이 쓰는 카테고리 1개를 특별히 설명해주세요. 
        고객에게 보여주는 리포트에는 고객의 자산이나 비율만 표시하고 기준점은 따로 설명하지 않습니다.
        전체적인 고객의 소비패턴을 분석하고 개선방향을 알려주세요.

        #### **종합 판단**
        [줄바꿈]
        고객의 전체적인 자산 현황, 소비 패턴, 저축 상황 등을 종합적으로 판단하여 한 줄 요약으로 정리해주고, 마지막에는 응원의 메세지를 함께 출력해주세요.

        리포트를 작성할 때 따뜻하고 희망적인 메시지를 중심으로, 고객이 긍정적인 변화를 추구할 수 있도록 도와주세요!
        리포트 작성 시 한 문장 마다 띄어쓰기해주세요. 
        """
        # OpenAI API 호출

        # openai.api_key = os.getenv('APIKEY')
        # response = openai.ChatCompletion.create(
        report_content = request.session.get('report_content', None)
        if request.method == 'POST' and not report_content:
            # OpenAI API 호출 또는 리포트 생성
            response = client.chat.completions.create(
                model="gpt-4", 
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            report_content = response.choices[0].message.content

            # 세션에 저장
            request.session['report_content'] = report_content
            request.session.modified = True  # 세션 변경 사항 저장을 강제

        if report_content is None:
            report_content = ""
        # JSON 직렬화된 데이터를 템플릿에 전달
        context = {
            'bar_data': json.dumps(bar_data, ensure_ascii=False),
            'average_data': json.dumps(average_values, ensure_ascii=False),
            'user_name': user_name,
            "report": report_content,
            'sorted_categories_json' : sorted_categories_json,
            'mydata_assets_json' : mydata_assets_json,
        }
        return render(request, 'report_origin.html', context)

    # except Exception as e:
    #     return render(request, "error.html", {"message": str(e)})

    except UserProfile.DoesNotExist:
        return render(request, 'report_origin.html', {'error': '사용자 정보를 찾을 수 없습니다.'})

@csrf_exempt
def log_to_elasticsearch(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # 세션에서 사용자 ID 가져오기
            customer_id = request.session.get('user_id')

            user_name = "사용자"  # 기본값 설정
            if customer_id:
                try:
                    # CustomerID로 UserProfile 조회
                    user = UserProfile.objects.get(CustomerID=customer_id)
                    user_name = user.username  # 사용자 이름 설정
                    customer_class = {
                        "Stageclass": user.Stageclass,  
                        "Inlevel": user.Inlevel,        
                    }

                    product_name = data.get('product_name')

                    timestamp = datetime.now().isoformat()
                except UserProfile.DoesNotExist:
                    pass  # 사용자가 없을 경우 기본값 유지
           

            # Elasticsearch에 저장할 데이터
            document = {
                'customer_id': customer_id,
                'data': {
                    'product_name': data.get('product_name', 'N/A'),
                    'bank': data.get('bank', 'N/A'),
                    'baser': data.get('baser', 'N/A'),
                    'maxir': data.get('maxir', 'N/A'),
                    'method': data.get('method', 'N/A'),
                },
                'customer_class' :customer_class,
                'timestamp': timestamp
            }

            # Elasticsearch에 데이터 저장
            es.index(index="ps_product_click_logs", document=document)

            return JsonResponse({"status": "success", "message": "Click logged to Elasticsearch"}, status=200)
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=400)

@login_required_session
def better_option(request):
    customer_id = request.session.get('user_id')  
    user_name = "사용자"  # 기본값 설정

    if customer_id:
        try:
            # CustomerID로 UserProfile 조회
            user = UserProfile.objects.get(CustomerID=customer_id)
            user_name = user.username  # 사용자 이름 설정
            accounts = MyDataDS.objects.filter(CustomerID=customer_id).values('AccountID','bank_name', 'balance','pname', 'ds_rate','end_date','dstype')
            today = timezone.now().date()
            accounts_list = [
                {
                    "AccountID": account['AccountID'],
                    "bank_name": account['bank_name'],
                    "balance": account['balance'],
                    "pname": account['pname'],
                    "ds_rate": float(account['ds_rate']),
                    "dstype": account['dstype'],
                    "end_date": account['end_date'],
                    "days_remaining": (account['end_date'] - today).days
                }
                for account in accounts
                if account['end_date'] > today  # 만기가 지나지 않은 상품만 포함
            ]
            if accounts_list:
                nearest_expiring = min(accounts_list, key=lambda x: x['days_remaining'])
            d_list = [account for account in accounts_list if account['dstype'] == 'd']
            s_list = [account for account in accounts_list if account['dstype'] == 's']

            # 각각의 리스트에서 만기일이 가장 가까운 상품 선택
            if d_list:
                nearest_d = min(d_list, key=lambda x: x['days_remaining'])
            if s_list:
                nearest_s = min(s_list, key=lambda x: x['days_remaining'])
        except UserProfile.DoesNotExist:
            pass  # 사용자가 없을 경우 기본값 유지
    final_recommend = request.session.get('final_recommend', [])
    deposit_recommend = request.session.get('deposit_recommend', [])

    # 각 추천 데이터에 로고 추가
    try:
        final_recommend = json.loads(final_recommend) if isinstance(final_recommend, str) else final_recommend
        deposit_recommend = json.loads(deposit_recommend) if isinstance(deposit_recommend, str) else deposit_recommend
    except json.JSONDecodeError:
        final_recommend = []
        deposit_recommend = []
    nearest_d_with_logo = (
        {**nearest_d, "logo": get_bank_logo(nearest_d.get("bank_name", ""))} if nearest_d else None
    )
    nearest_s_with_logo = (
        {**nearest_s, "logo": get_bank_logo(nearest_s.get("bank", ""))} if nearest_s else None
    )
    for item in deposit_recommend:
        logger.debug(f"Item: {item}")
        if 'dsid' not in item or not item['dsid']:
            logger.error(f"Invalid dsid in item: {item}")
    context = {
        'user_name': user_name,
        'final_recommend': final_recommend,  # 적금 Top 5
        'deposit_recommend': deposit_recommend,  # 예금 Top 5
        'nearest_d' : nearest_d_with_logo,
        'nearest_s' : nearest_s_with_logo,
    }

    return render(request, 'better_options.html',context)

@login_required_session
def d_detail(request,dsid):
    customer_id = request.session.get('user_id')  
    user_name = "사용자"  # 기본값 설정
    if customer_id:
        try:
            # CustomerID로 UserProfile 조회
            user = UserProfile.objects.get(CustomerID=customer_id)
            user_name = user.username  # 사용자 이름 설정
        except UserProfile.DoesNotExist:
            pass  # 사용자가 없을 경우 기본값 유지

    try:
        product = DProduct.objects.get(dsid=dsid)
        index_name = "d_products"  # 인덱스 이름
        index_name_2 = "d_products_tip"
        product_img = get_bank_logo(product.bank)
        # Elasticsearch 검색 쿼리
        query = {
            "_source": ["dsid", "context"],  # 필요한 필드만 가져옴
            "query": {
                "term": {
                    "dsid": dsid  # dsid 값과 정확히 매칭
                }
            }
        }

        try:

            # Elasticsearch 검색
            response = es.search(index=index_name, body=query)
            hits = response.get("hits", {}).get("hits", [])

            if hits:
                # context 값만 추출
                context_value = hits[0]["_source"]["context"]

            # Elasticsearch 검색
            response = es.search(index=index_name_2, body=query)
            hits = response.get("hits", {}).get("hits", [])

            if hits:
                # context 값만 추출
                context_value_tip = hits[0]["_source"]["context"]

                
            else:
                return JsonResponse({"status": "error", "message": "데이터를 찾을 수 없습니다."}, status=404)

        except Exception as e:
            # 에러 처리
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    except DProduct.DoesNotExist:
        return render(request, 'error.html', {'message': '해당 상품을 찾을 수 없습니다.'})
    
 
    context = {
        'product': product,
        'context_value' : context_value,
        'context_value_tip':context_value_tip,
        'user_name' : user_name,
        'product_img': product_img,
    }

    return render(request, 'd_detail.html',context)

@login_required_session
def s_detail(request, dsid):
    # s_product에서 먼저 검
    customer_id = request.session.get('user_id')  
    user_name = "사용자"  # 기본값 설정색
    if customer_id:
        try:
            # CustomerID로 UserProfile 조회
            user = UserProfile.objects.get(CustomerID=customer_id)
            user_name = user.username  # 사용자 이름 설정
        except UserProfile.DoesNotExist:
            pass  # 사용자가 없을 경우 기본값 유지
    try:
        product = SProduct.objects.get(DSID=dsid)
        product_img = get_bank_logo(product.bank_name)
    except SProduct.DoesNotExist:
        return render(request, 'error.html', {'message': '해당 상품을 찾을 수 없습니다.'})

    # 적절한 데이터를 템플릿으로 전달
    context = {
        'product': product,
        'user_name' : user_name,
        'product_img': product_img,
    }
    return render(request, 's_detail.html', context)

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

def search(request):
    return render(request, 'search.html')