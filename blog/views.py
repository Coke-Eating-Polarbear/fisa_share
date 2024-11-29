from django.shortcuts import render, redirect,get_object_or_404 # type: ignore
from django.contrib.auth import authenticate, login,logout # type: ignore
from django.utils import timezone # type: ignore
from datetime import timedelta
import matplotlib.pyplot as plt # type: ignore
import base64
import io
from matplotlib import font_manager, rc # type: ignore
from django.contrib import messages # type: ignore
from .forms import UserProfileForm
from blog.models import UserProfile,Recommend, Wc, News, Favorite, Average,card, MyDataAsset, MyDataDS, MyDataPay,SpendAmount, DProduct, SProduct, SpendFreq  # UserProfile 모델도 가져옵니다
from django.contrib.auth.hashers import check_password# type: ignore
from django.views.decorators.http import require_POST# type: ignore
from django.http import HttpResponse,JsonResponse# type: ignore
from django.db.models import F # type: ignore
import random
import logging
from .logging import *
from elasticsearch import Elasticsearch # type: ignore
from django.views.decorators.csrf import csrf_exempt # type: ignore
import json
import os
from dotenv import load_dotenv # type: ignore
from collections import defaultdict
from accounts.views import map_person
from .utils import income_model
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from joblib import load
import numpy as np
from django.conf import settings
from openai import OpenAI
from django.db.models import Q
from django.core.serializers.json import DjangoJSONEncoder

es = Elasticsearch([os.getenv('ES')])  # Elasticsearch 설정
load_dotenv() 
# openai.api_key = os.getenv('APIKEY')
client = OpenAI()

rc('font', family='Malgun Gothic')
logger = logging.getLogger(__name__)
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
    reverse_data = reverse_mapping_with_age(user.stageclass, age)

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
        user.stageclass = updated_stage_class
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

    if customer_id:
        try:
            # CustomerID로 UserProfile 조회
            user = UserProfile.objects.get(CustomerID=customer_id)
            user_name = user.username  # 사용자 이름 설정

            # MyDataDS 모델에서 해당 CustomerID에 연결된 계좌 정보 가져오기
            accounts = MyDataDS.objects.filter(CustomerID=customer_id).values('AccountID', 'balance')
        except UserProfile.DoesNotExist:
            pass  # 사용자가 없을 경우 기본값 유지

    context = {
        'user_name': user_name,  # 사용자 이름
        'accounts': accounts,   # 계좌 정보 리스트
    }
    return render(request, 'mypage.html', context)

def fetch_sql_processed_data():
    """
    SQL에서 전처리된 데이터를 가져오는 함수.
    Returns:
        DataFrame: SQL에서 처리된 데이터를 Pandas DataFrame으로 반환
    """
    db_config = {
        'host': '118.67.131.22:3306',
        'user': 'fisaai',
        'password': 'woorifisa3!W',
        'database': 'manduck'
    }
    db_connection = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
    engine = create_engine(db_connection)

    query = """
    SELECT 
        Pyear,
        Pmonth,
        Bizcode,
        SUM(Price) AS TotalPrice,
        SUM(SUM(Price)) OVER (PARTITION BY Pyear, Pmonth) AS TotalSpending,
        SUM(Price) * 1.0 / SUM(SUM(Price)) OVER (PARTITION BY Pyear, Pmonth) AS Ratio
    FROM mydata_pay
    GROUP BY Pyear, Pmonth, Bizcode
    ORDER BY Pyear, Pmonth, Bizcode;
    """
    df = pd.read_sql(query, engine)

    # Pivot 변환: Bizcode를 열로 만들고 각 Ratio 값을 채움
    pivot_data = df.pivot(index=['Pyear', 'Pmonth'], columns='Bizcode', values='Ratio').fillna(0)

    # TotalSpending 추가
    pivot_data['TotalSpending'] = df.drop_duplicates(subset=['Pyear', 'Pmonth'])[['Pyear', 'Pmonth', 'TotalSpending']].set_index(['Pyear', 'Pmonth'])

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

    # 디버깅: 가장 최근 데이터 확인
    print(f"가장 최근 데이터 (모델 입력 전):\n{most_recent_data}")

    # Series에서 모델 입력 데이터 생성
    model_input = most_recent_data.drop(labels=['TotalSpending'], errors='ignore')

    # 디버깅: 모델 입력 데이터 확인
    print(f"모델 입력 데이터 (가장 최근 데이터):\n{model_input}")

    # 모델 로드 및 예측
    model = load('rfm_Consumption_prediction.joblib')
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

def senter():
    """
    메인 함수: 데이터 처리, 예측, 출력 수행
    """
    print("SQL에서 전처리된 데이터를 가져옵니다...")
    preprocessed_data = fetch_sql_processed_data()
    print("Preprocessed Data Columns:", preprocessed_data.columns)

    print("저장된 모델의 입력 형식을 확인합니다...")
    model = os.path.join(settings.BASE_DIR, 'models', 'rfm_Consumption_prediction.joblib')
    model_features = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else preprocessed_data.columns.drop('TotalSpending')

    print("다음 달 예측 결과:")
    next_month_prediction = predict_next_month(preprocessed_data, model_features)
    print(f"연도: {next_month_prediction.name[0]}, 월: {next_month_prediction.name[1]}")
    print(next_month_prediction)
    return next_month_prediction

@login_required_session
def spending_mbti(request):
    customer_id = request.session.get('user_id')  
    user_name = "사용자"
    if customer_id:
        try:
            # CustomerID로 UserProfile 조회
            user = UserProfile.objects.get(CustomerID=customer_id)
            user_name = user.username  # 사용자 이름 설정

        except UserProfile.DoesNotExist:
            pass  # 사용자가 없을 경우 기본값 유지

    ## 소비예측 모델 넣기
    # MySQL 연결 정보

    pd.options.display.float_format = '{:,.2f}'.format
    next_month_prediction = None
    if __name__ == "__main__":
        prediction= senter()
        # JSON 형식으로 변환
        next_month_prediction = json.dumps(prediction)

    context = {
        'user_name': user_name,
        'next_month_prediction' : next_month_prediction, # 다음 달 소비 예측 json.
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
        DProduct.objects.filter(dsid__in=recommended_dsid_list['dproduct']).values('dsname', 'bank', 'baser', 'maxir')
    ) + [
        {
            'dsname': sp['product_name'],
            'bank': sp['bank_name'],
            'baser': sp['base_rate'],
            'maxir': sp['max_preferential_rate']
        }
        for sp in SProduct.objects.filter(DSID__in=recommended_dsid_list['sproduct']).values('product_name', 'bank_name', 'base_rate', 'max_preferential_rate')
    ]

    # 랜덤 상품 추가
    if recommended_count < 5:
        remaining_count = 5 - recommended_count
        random_dproducts = DProduct.objects.exclude(dsid__in=recommended_dsid_list['dproduct']).order_by('?')[:remaining_count]
        random_sproducts = SProduct.objects.exclude(DSID__in=recommended_dsid_list['sproduct']).order_by('?')[:remaining_count]

        random_product_details = list(random_dproducts.values('dsname', 'bank', 'baser', 'maxir')) + [
            {
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
    current_year = datetime.datetime.now().year
    age = current_year - birth_year
    cluster = assign_cluster(user.Stageclass, user.sex, age)

    for i in cluster:
        filtered_df = cluster_savings[cluster_savings['cluster1'] == i]
        if not filtered_df.empty:
            sorted_df = filtered_df.sort_values(by=['max_preferential_rate', 'base_rate'], ascending=[False, False])
            if not sorted_df.empty:
                top_result = sorted_df.head(2)
                final_result = pd.concat([final_result, top_result], ignore_index=True)

    # 적금 최종 추천 3개로 제한
    final_recommend_json = final_result.head(3)[["product_name", "bank_name", "max_preferential_rate", "base_rate", "signup_method"]].to_dict(orient='records')
    

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

    final_recommendations = pd.concat(filtered_results, ignore_index=True)
    top2 = final_recommendations.sort_values(by='maxir', ascending=False).head(2)
    deposit_recommend_json = top2.to_dict(orient='records')
    request.session['final_recommend'] = final_recommend_json
    request.session['deposit_recommend'] = deposit_recommend_json
    print("Session Final Recommend:", request.session.get('final_recommend'))
    print("Session Deposit Recommend:", request.session.get('deposit_recommend'))
    # 최종 데이터 전달
    context = {
        'product_details': product_details,
        'image_base64': image_base64,
        'news_entries': news_entries,
        'user_name': user_name,
        'final_recommend': final_recommend_json,  # 적금 Top 3
        'deposit_recommend': deposit_recommend_json  # 예금 Top 2
    }

    return render(request, 'loginmain.html', context)

@login_required_session
def info(request):
    # 세션에서 CustomerID 가져오기
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
    
    # POST 요청일 경우 info1의 goal과 saving_method를 세션에 저장 후 리디렉션
    if request.method == 'POST':
        biggoal = request.POST.get('biggoal')
        goal = request.POST.get('goal')
        saving_method = request.POST.get('saving_method')
        period = request.POST.get('period')
        amount = request.POST.get('amount')
        bank_option = request.POST.get('bank_option')
        selected_preferences = request.POST.getlist('preferences')
        
        request.session['biggoal'] = biggoal
        request.session['selected_preferences'] = selected_preferences
        request.session['bank_option'] = bank_option
        request.session['period'] = period
        request.session['amount'] = amount
        request.session['goal'] = goal
        request.session['saving_method'] = saving_method
        print(biggoal,selected_preferences,bank_option,period,amount,goal,saving_method)
        return redirect('top5')
    
    # GET 요청일 경우 템플릿 렌더링
    return render(request, 'savings_info.html', context)

@login_required_session
def top5(request):
    customer_id = request.session.get('user_id')  
    user_name = "사용자"  # 기본값 설정
    top5_products = []  # 추천 상품 리스트 초기화

    if customer_id:
        try:
            # CustomerID로 UserProfile 조회
            user = UserProfile.objects.get(CustomerID=customer_id)
            user_name = user.username  # 사용자 이름 설정

            # Favorite 테이블에서 사용자와 관련된 DSID 가져오기
            favorites = Favorite.objects.filter(CustomerID=user).select_related('content_type')

            # Favorite에 등록된 상품 중 상위 5개 가져오기
            top5_products = favorites[:5]  # 필요한 로직에 따라 상위 5개만 선택
        except UserProfile.DoesNotExist:
            pass  # 사용자가 없을 경우 기본값 유지
    final_recommend = request.session.get('final_recommend')
    deposit_recommend = request.session.get('deposit_recommend')
    print("Final Recommend:", final_recommend)
    print("Deposit Recommend:", deposit_recommend)
    context = {
        'user_name': user_name,
        'top5_products': top5_products,
        'final_recommend': final_recommend,  # 적금 Top 3
        'deposit_recommend': deposit_recommend  # 예금 Top 2
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
def favorite(request):
    customer_id = request.session.get('user_id')
    user_name = "사용자"
    top5_products = []

    if customer_id:
        try:
            # CustomerID로 UserProfile 조회
            user = UserProfile.objects.get(CustomerID=customer_id)
            user_name = user.username

            # Favorite 테이블에서 사용자와 관련된 DSID 가져오기
            top5_products = Favorite.objects.filter(CustomerID=user).select_related('DSID')[:5]
        except UserProfile.DoesNotExist:
            pass

    context = {
        'user_name': user_name,
        'top5_products': top5_products,
    }

    return render(request, 'favorites.html', context)

@csrf_exempt
def add_favorite(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            product_id = data.get("product_id")  # 제품 ID
            product_type = data.get("product_type")  # 제품 타입 (dproduct or sproduct)
            customer_id = request.session.get('user_id')  # 세션에서 사용자 ID 가져오기

            if customer_id and product_id and product_type:
                user = UserProfile.objects.get(CustomerID=customer_id)

                if product_type == "dproduct":  # 예금인 경우
                    dproduct = DProduct.objects.get(dsid=product_id)
                    Favorite.objects.get_or_create(CustomerID=user, dproduct=dproduct)
                elif product_type == "sproduct":  # 적금인 경우
                    sproduct = SProduct.objects.get(DSID=product_id)
                    Favorite.objects.get_or_create(CustomerID=user, sproduct=sproduct)
                else:
                    return JsonResponse({"status": "error", "message": "Invalid product type"}, status=400)

                return JsonResponse({"status": "success"})

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)

    return JsonResponse({"status": "error"}, status=400)

@csrf_exempt
def remove_favorite(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            product_id = data.get("product_id")  # 제품 ID
            product_type = data.get("product_type")  # 제품 타입 (dproduct or sproduct)
            customer_id = request.session.get('user_id')  # 세션에서 사용자 ID 가져오기

            if customer_id and product_id and product_type:
                user = UserProfile.objects.get(CustomerID=customer_id)

                if product_type == "dproduct":  # 예금인 경우
                    dproduct = DProduct.objects.get(dsid=product_id)
                    Favorite.objects.filter(CustomerID=user, dproduct=dproduct).delete()
                elif product_type == "sproduct":  # 적금인 경우
                    sproduct = SProduct.objects.get(DSID=product_id)
                    Favorite.objects.filter(CustomerID=user, sproduct=sproduct).delete()
                else:
                    return JsonResponse({"status": "error", "message": "Invalid product type"}, status=400)

                return JsonResponse({"status": "success"})

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)

    return JsonResponse({"status": "error"}, status=400)

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
        print("Customer ID:", customer_id)  # 디버깅용 출력
        print("User Data:", user)
        cnow = datetime.now()
        current_month = cnow.strftime("%Y-%m")
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
        print("Average Data:", average_data)  # 디버깅용 출력

        # MyData에서 고객 데이터 조회
        if not user_asset_data:
            raise ValueError(f"사용자 데이터를 찾을 수 없습니다. (Customer ID: {customer_id})")
        if not spend_amount:
            raise ValueError(f"사용자 데이터를 찾을 수 없습니다. (Customer ID: {customer_id})")
        if not spend_freq:
            raise ValueError(f"사용자 데이터를 찾을 수 없습니다. (Customer ID: {customer_id})")
        print("User Financial Data-usd:", user_asset_data)  # 디버깅용 출력
        print("User Financial Data-sa:", spend_amount)  # 디버깅용 출력
        print("User Financial Data-sf:", spend_freq)  # 디버깅용 출력

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
        리포트는 '만덕이'가 고객에게 이야기하듯 친근하고 귀여운 말투로 작성해주세요. 만덕이는 긍정적이고 따뜻하게 응원해주는 친구로, 고객이 스스로를 격려하며 개선 방향을 이해할 수 있도록 설명합니다.
        예를 들어, "오! 정말 잘하고 있어요! 조금만 더 이렇게 하면 완벽할 거예요"처럼 친절하고 귀여운 말투를 사용하세요.
        리포트를 작성할 때 아래 기준을 참고해 고객의 상황을 자세히 분석해주세요.
        리포트 작성 시 한 줄 마다 마크다운 문법으로 띄어쓰기, 줄바꿈 표시도 함께 넣어주세요.
        아래는 리포트 구성 항목입니다.

        ### 입력 데이터:
        - 고객의 기본 데이터:
        - CustomerID: {customer_id}
        - 현재 연월: {last_month_str}
        - 해당 그룹: {user.Stageclass}
        - 소득 분위: {user.Inlevel}

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
        #### **1. 자산 현황 분석**
        - **순자산 비교:**
        - 고객의 순자산이 평균 그룹의 순자산보다 높은지 낮은지를 비교하고, 그에 따른 긍정적이거나 개선을 위한 메시지를 전달해주세요.
        - 예: "현재 순자산이 평균보다 많아요, 덕! 너무 잘하고 있어요~" 또는 "조금만 더 노력하면 평균을 넘어설 수 있을 거예요!"
        - **자산 비율 분포:**
        - 금융자산, 부동산자산, 기타자산의 비율이 전체 자산에서 차지하는 비중을 설명하고, 그룹 평균과 비교하여 각각 많은지 적은지를 알려주세요.
        - 예: "금융자산 비율이 평균보다 조금 적은데요, 덕. 이 부분을 조금만 더 늘려보면 좋을 것 같아요~"

        #### **2. 자산 관리 지표 분석**
        - **비유동자산 비율:**
        - 50 이하가 이상적임을 알려주고, 고객의 비율이 적정한지 판단해주세요.
        - 예: "비유동자산 비율이 50 이하로 아주 건강한 상태예요~ 덕!"
        - 만약 50 이상이라면, "조금 높지만 괜찮아요! 이 비율을 낮추는 방법에 대해 같이 고민해봐요~ 덕!"
        - **부채 비율:**
        - 40 이하(이상적), 40-80(주의 필요), 80 이상(위험)을 기준으로 상태를 진단하고, 개선 방향을 제시하세요.
        - 예: "부채 비율이 40 이하로 정말 안정적이에요, 덕!" 또는 "부채 비율이 조금 높아요~ 덕. 조금씩 줄여보는 건 어때요?"
        - **유동성 비율:**
        - 2 이상(매우 양호), 1.5-2(양호), 1.0-1.5(주의 필요), 1.0 이하(위험)의 기준으로 건강 상태를 분석하고 응원의 메시지를 전달해주세요.

        #### **3. 저축 및 소비 습관 분석**
        - **월저축률:**
        - 월소득 대비 저축률이 10-20% 미만이면 위험, 10-20%는 최소 충족, 50-60%가 이상적임을 기준으로 평가하고, 고객의 저축 습관에 대해 칭찬하거나 개선 방향을 제안하세요.
        - **총자산저축률:**
        - 20-30%가 적절한 수준이며, 이를 기준으로 고객의 상황을 설명하고 개선 방법을 제시해주세요.
        - **소득 대비 자산 비율:**
        - 3-5가 양호, 5 이상은 아주 좋은 수준, 3 미만은 자산 형성이 필요한 상태임을 기준으로 고객의 자산 형성 상태를 판단하세요.
        - 예: "와~ 소득 대비 자산 비율이 5 이상이에요! 정말 훌륭해요, 덕!"
        - 또는 "3 미만이지만 괜찮아요. 천천히 자산을 늘려나가면 더 좋아질 거예요!"

        #### **4. 소비 분석**
        - **평균과의 소비 비교:**
        - 평균보다 많이 지출하거나 적게 지출하는지를 알려주고, 이에 대한 피드백을 주세요.
        - 예: "평균보다 적게 쓰고 있어요. 저축도 잘하고 있는 모습이에요~ 덕!"
        - 또는 "조금 많이 쓰는 경향이 있지만, 적당히 조절하면 완벽할 거예요~ 덕!"
        - **카테고리별 소비 차이 분석:**
        - 소비 카테고리별 비중을 분석해 가장 소비가 많은 상위 5개 카테고리를 나열하고, 가장 많이 쓰는 카테고리 1개를 특별히 설명해주세요.
        - 예: "가장 많이 소비한 항목은 '식비'예요! 하지만 괜찮아요, 덕. 맛있는 음식을 즐기면서 절약하는 방법도 함께 찾아볼 수 있어요!"

        #### **5. 종합 판단**
        - **지출 비율 분석:**
        - 개인 소득 대비 지출 비중이 50% 이상이면 건강한 상태임을 알려주고, 이에 대한 칭찬이나 개선 방향을 제시하세요.

        리포트를 작성할 때 따뜻하고 희망적인 메시지를 중심으로, 고객이 긍정적인 변화를 추구할 수 있도록 도와주세요!
        """
        # OpenAI API 호출

        # openai.api_key = os.getenv('APIKEY')
        # response = openai.ChatCompletion.create(
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
        print(report_content)


        # 데이터 준비
        bar_data = {
            '총자산': user_asset_data.total,
            '현금자산': user_asset_data.financial,
            '수입': user_asset_data.monthly_income,
            '지출': abs(spend_amount.TotalAmount)
        }
        print("Debug: bar_data contents:", bar_data)
        average_values = {
            '총자산': (average_data.asset + average_data.finance),
            '현금자산': average_data.finance,
            '수입': average_data.income,
            '지출': average_data.spend
        }
        print("Debug: bar_data contents:", average_values)


        # JSON 직렬화된 데이터를 템플릿에 전달
        context = {
            'bar_data': json.dumps(bar_data, ensure_ascii=False),
            'average_data': json.dumps(average_values, ensure_ascii=False),
            'user_name': user_name,
            "report": report_content,
        }

        return render(request, 'report_origin.html', context)

    # except Exception as e:
    #     return render(request, "error.html", {"message": str(e)})

    except UserProfile.DoesNotExist:
        print("UserProfile 데이터가 없습니다.")  # 디버깅용
        return render(request, 'report_origin.html', {'error': '사용자 정보를 찾을 수 없습니다.'})
    # except Exception as e:
    #     print("에러 발생:", str(e))  # 디버깅용
    #     return render(request, 'report_origin.html', {'error': str(e)})