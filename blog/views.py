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
from blog.models import UserProfile,Recommend, Wc, News, Favorite, Average,card, MyDataAsset, MyDataDS, MyDataPay,SpendAmount, DProduct, SProduct  # UserProfile 모델도 가져옵니다
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
import datetime
import pandas as pd
from sqlalchemy import create_engine
from joblib import load
import numpy as np


es = Elasticsearch([os.getenv('ES')])  # Elasticsearch 설정
load_dotenv() 
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
    db_config = {
        'host': '118.67.131.22:3306',
        'user': 'fisaai',
        'password': 'woorifisa3!W',
        'database': 'manduck'
    }

    pd.options.display.float_format = '{:,.2f}'.format

    def fetch_sql_processed_data():
        """
        SQL에서 전처리된 데이터를 가져오는 함수.
        Returns:
            DataFrame: SQL에서 처리된 데이터를 Pandas DataFrame으로 반환
        """
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

    def main():
        """
        메인 함수: 데이터 처리, 예측, 출력 수행
        """
        print("SQL에서 전처리된 데이터를 가져옵니다...")
        preprocessed_data = fetch_sql_processed_data()
        print("Preprocessed Data Columns:", preprocessed_data.columns)

        print("저장된 모델의 입력 형식을 확인합니다...")
        model = load('rfm_Consumption_prediction.joblib')
        model_features = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else preprocessed_data.columns.drop('TotalSpending')

        print("다음 달 예측 결과:")
        next_month_prediction = predict_next_month(preprocessed_data, model_features)
        print(f"연도: {next_month_prediction.name[0]}, 월: {next_month_prediction.name[1]}")
        print(next_month_prediction)
        return next_month_prediction


    if __name__ == "__main__":
        prediction= main()
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
            'dsname': sp['ProductName'],
            'bank': sp['BankName'],
            'baser': sp['BaseRate'],
            'maxir': sp['MaxPreferentialRate']
        }
        for sp in SProduct.objects.filter(DSID__in=recommended_dsid_list['sproduct']).values('ProductName', 'BankName', 'BaseRate', 'MaxPreferentialRate')
    ]

    # 랜덤 상품 추가
    if recommended_count < 5:
        remaining_count = 5 - recommended_count
        random_dproducts = DProduct.objects.exclude(dsid__in=recommended_dsid_list['dproduct']).order_by('?')[:remaining_count]
        random_sproducts = SProduct.objects.exclude(DSID__in=recommended_dsid_list['sproduct']).order_by('?')[:remaining_count]

        random_product_details = list(random_dproducts.values('dsname', 'bank', 'baser', 'maxir')) + [
            {
                'dsname': sp.ProductName,
                'bank': sp.BankName,
                'baser': sp.BaseRate,
                'maxir': sp.MaxPreferentialRate
            }
            for sp in random_sproducts
        ]

        product_details = recommended_product_details + random_product_details
    else:
        product_details = recommended_product_details

    # 중복 제거 및 최대 5개 제한
    unique_product_details = {p['dsname']: p for p in product_details if p['dsname']}.values()
    product_details = list(unique_product_details)[:5]

    # 적금 추천 상품 처리
    cluster_savings = pd.read_csv('C:/Users/audwn/ITstudy/final_project/final/blog/cluster_savings_updated.csv')
    final_result = pd.DataFrame(columns=cluster_savings.columns)

    def assign_cluster(stage_class, sex, age):
        if stage_class == 0:
            if sex == 'M' and age in [19, 20, 21]:
                return [5, 2, 1, 4]
            else:
                return [0, 1, 4]
        else:
            return [1, 4]

    birth_year = user.Birth.year  # 주민번호 앞자리로 연도 추출
    current_year = datetime.datetime.now().year
    age = current_year - birth_year
    cluster = assign_cluster(user.Stageclass, user.sex, age)

    for i in cluster:
        filtered_df = cluster_savings[cluster_savings['cluster1'] == i]
        if not filtered_df.empty:
            sorted_df = filtered_df.sort_values(by=['최고우대금리', '기본금리'], ascending=[False, False])
            if not sorted_df.empty:
                top_result = sorted_df.head(2)
                final_result = pd.concat([final_result, top_result], ignore_index=True)

    # 적금 최종 추천 3개로 제한
    final_recommend_json = final_result.head(3)[["상품명", "은행명", "최고우대금리", "기본금리", "가입방법"]].to_dict(orient='records')

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
        filtered_deposits_query = DProduct.objects.filter(cluster=cluster).values('dsid', 'name', 'bank', 'baser', 'maxir')
        filtered_results.append(pd.DataFrame(filtered_deposits_query))

    final_recommendations = pd.concat(filtered_results, ignore_index=True)
    top2 = final_recommendations.sort_values(by='maxir', ascending=False).head(2)
    deposit_recommend_json = top2.to_dict(orient='records')

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

    context = {
        'user_name': user_name,
        'top5_products': top5_products,
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
        now = datetime.now()
        current_month = now.strftime("%Y-%m")
        # Average 테이블에서 고객 소득분위 기준 데이터 조회
        average_data = Average.objects.filter(
            stageclass=user.stageclass,
            inlevel=user.inlevel
        ).first()

        if not average_data:
            raise ValueError(f"소득 분위 데이터가 없습니다. (Stage Class: {user.stageclass}, Inlevel: {user.inlevel})")
        print("Average Data:", average_data)  # 디버깅용 출력

        # MyData에서 고객 데이터 조회
        user_asset_data = MyDataAsset.objects.filter(CustomerID=customer_id).first()
        if not user_asset_data:
            raise ValueError(f"사용자 데이터를 찾을 수 없습니다. (Customer ID: {customer_id})")
        user_spend_data = SpendAmount.objects.filter(CustomerID=customer_id).first()
        if not user_spend_data:
            raise ValueError(f"사용자 데이터를 찾을 수 없습니다. (Customer ID: {customer_id})")
        print("User Financial Data:", user_asset_data)  # 디버깅용 출력
        print("User Financial Data:", user_spend_data)  # 디버깅용 출력

        # 데이터 준비
        bar_data = {
            '총자산': user_asset_data.total,
            '현금자산': user_asset_data.financial,
            '수입': user_asset_data.income,
            '지출': abs(user_spend_data.TotalAmount)
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
        }

        return render(request, 'report_origin.html', context)

    except UserProfile.DoesNotExist:
        print("UserProfile 데이터가 없습니다.")  # 디버깅용
        return render(request, 'report_origin.html', {'error': '사용자 정보를 찾을 수 없습니다.'})
    except Exception as e:
        print("에러 발생:", str(e))  # 디버깅용
        return render(request, 'report_origin.html', {'error': str(e)})