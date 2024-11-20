from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.utils import timezone
from datetime import timedelta
import base64
from django.contrib import messages
from .forms import UserProfileForm  # UserProfileForm을 가져옵니다
from blog.models import UserProfile,Recommend, DsProduct, Wc, News, Favorite  # UserProfile 모델도 가져옵니다
from django.contrib.auth.hashers import check_password
from django.views.decorators.http import require_POST
from django.http import HttpResponse,JsonResponse
from django.db.models import F
import random
import logging
from .logging import *
from elasticsearch import Elasticsearch
from django.views.decorators.csrf import csrf_exempt
import json
import os
from dotenv import load_dotenv
from collections import defaultdict
load_dotenv()

def login_required_session(view_func):
    """
    세션에 'user_id'가 없을 경우 로그인 페이지로 리디렉션하는 데코레이터.
    """
    def wrapper(request, *args, **kwargs):
        # 세션에 'user_id'가 없으면 로그인 페이지로 리디렉션
        if not request.session.get('user_id'):
            return redirect('login')  # 'login' URL로 이동
        # 'user_id'가 있으면 원래의 뷰 함수 실행
        return view_func(request, *args, **kwargs)
    return wrapper

logger = logging.getLogger(__name__)



@login_required_session
def mypage(request):
    customer_id = request.session.get('user_id')  
    user_name = "사용자"
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
    return render(request, 'mypage.html',context)


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

    context = {
        'user_name': user_name,
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
    yesterday = today - timedelta(days=1)
    # 오늘 날짜의 이미지 데이터 가져오기
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
            news_entries.append({
                'title': news.title,
                'summary': news.summary,
                'url': news.url  # URL 추가
            })
    # try:
    #     wc_entry = Wc.objects.filter(date=yesterday).first()
    #     if wc_entry is not None and wc_entry.image:
    #         image_base64 = base64.b64encode(wc_entry.image).decode('utf-8')  # base64 인코딩
    #     else:
    #         image_base64 = None  # 이미지가 없거나 wc_entry가 None일 경우 처리
    # except Exception as e:
    # # 예상치 못한 오류를 잡아내기 위한 처리
    #     image_base64 = None
    #     print(f"An error occurred: {e}")


    # news_entries = News.objects.filter(ndate=yesterday, summary__isnull=False).values('title', 'summary', 'url')

    # CustomerID가 세션에 없으면 로그인 페이지로 리디렉션
    if not customer_id:
        return redirect('login')  # 로그인 페이지 URL로 수정 필요

    # 추천 테이블에서 CustomerID에 해당하는 추천 상품 가져오기
    recommended_products = Recommend.objects.filter(CustomerID=customer_id).values('DSID')
    recommended_count = recommended_products.count()
    recommended_dsid_list = []

    # 추천된 상품이 있을 경우 해당 dsid로 ds_product에서 정보 가져오기
    recommended_product_details = []
    if recommended_count > 0:
        recommended_dsid_list = [rec['DSID'] for rec in recommended_products]
        recommended_product_details = list(
            DsProduct.objects.filter(DSID=recommended_dsid_list)
            .values('dsname', 'bank', 'baser', 'maxir')
        )

    # 추천 상품이 5개 미만인 경우, 부족한 개수만큼 중복되지 않게 ds_product에서 랜덤한 상품 가져오기
    if recommended_count < 5:
        remaining_count = 5 - recommended_count
        random_products = DsProduct.objects.exclude(dsid__in=recommended_dsid_list).order_by('?')[:remaining_count]
        random_product_details = random_products.values('dsname', 'bank', 'baser', 'maxir')

        # 추천 상품 + 랜덤 상품을 결합하여 총 5개의 상품을 보여줍니다.
        product_details = recommended_product_details + list(random_product_details)
    else:
        product_details = recommended_product_details

    # 로그에 product_details 출력
    logger.info("Product details: %s", product_details)

    # 모든 데이터를 하나의 context 딕셔너리로 전달
    context = {
        'product_details': product_details,
        'image_base64': image_base64,
        'news_entries': news_entries,
    }
    
    return render(request, 'loginmain.html', context)


@login_required_session
def info1(request):
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
        goal = request.POST.get('goal')
        saving_method = request.POST.get('saving_method')
        
        # 세션에 선택한 데이터 저장
        request.session['info1_goal'] = goal
        request.session['info1_saving_method'] = saving_method

        # 'savings_info2' 페이지로 리디렉션
        return redirect('info2')
    
    # GET 요청일 경우 템플릿 렌더링
    return render(request, 'savings_info1.html', context)


@login_required_session
def info2(request):
    customer_id = request.session.get('user_id')  
    user_name = "사용자"  # 기본값 설정

    if customer_id:
        try:
            # CustomerID로 UserProfile 조회
            user = UserProfile.objects.get(CustomerID=customer_id)
            user_name = user.username  # 사용자 이름 설정
        except UserProfile.DoesNotExist:
            pass  # 사용자가 없을 경우 기본값 유지

    if request.method == 'POST':
        # POST 요청에서 info2의 폼 데이터 가져오기
        goal = request.POST.get('goal')
        period = request.POST.get('period')
        amount = request.POST.get('amount')

        # 세션에 데이터 저장
        request.session['info2_goal'] = goal
        request.session['info2_period'] = period
        request.session['info2_amount'] = amount

        # 다음 페이지로 리디렉션
        return redirect('info3')  # 다음 페이지 URL 이름에 맞게 수정

    context = {
        'user_name': user_name,
    }
    return render(request, 'savings_info2.html', context)


@login_required_session
def info3(request):
    customer_id = request.session.get('user_id')
    user_name = "사용자"  # 기본값 설정

    if customer_id:
        try:
            # CustomerID로 UserProfile 조회
            user = UserProfile.objects.get(CustomerID=customer_id)
            user_name = user.username  # 사용자 이름 설정
        except UserProfile.DoesNotExist:
            pass  # 사용자가 없을 경우 기본값 유지

    if request.method == 'POST':
        # POST 요청에서 금융권 옵션 가져오기
        bank_option = request.POST.get('bank_option')
        
        # 세션에 금융권 옵션 저장 (겹치지 않도록 명확한 키 이름 사용)
        request.session['info3_bank_option'] = bank_option

        # 다음 페이지로 리디렉션
        return redirect('info4')  # 다음 페이지 URL 이름에 맞게 수정

    context = {
        'user_name': user_name,
    }
    return render(request, 'savings_info3.html', context)


@login_required_session
def info4(request):
    customer_id = request.session.get('user_id')
    user_name = "사용자"  # 기본값 설정

    if customer_id:
        try:
            # CustomerID로 UserProfile 조회
            user = UserProfile.objects.get(CustomerID=customer_id)
            user_name = user.username  # 사용자 이름 설정
        except UserProfile.DoesNotExist:
            pass  # 사용자가 없을 경우 기본값 유지

    if request.method == 'POST':
        # POST 요청에서 선택한 우대사항을 리스트로 가져옵니다.
        selected_preferences = request.POST.getlist('preferences')
        
        # 세션에 우대사항을 저장합니다.
        request.session['selected_preferences'] = selected_preferences

        # 다음 페이지로 리디렉션합니다.
        return redirect('top5')

    context = {
        'user_name': user_name,
    }
    # GET 요청 시 페이지 렌더링
    return render(request, 'savings_info4.html', context)


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
            favorites = Favorite.objects.filter(CustomerID=user).select_related('DSID')

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

es = Elasticsearch([os.getenv('ES')])  # Elasticsearch 설정

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


#----------------------------------------------------------------------
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
            product_id = data.get("product_id")  # DSID
            customer_id = request.session.get('user_id')  # 세션에서 사용자 ID 가져오기

            if customer_id and product_id:
                user = UserProfile.objects.get(CustomerID=customer_id)
                ds_product = DsProduct.objects.get(dsid=product_id)

                # Favorite에 추가
                Favorite.objects.get_or_create(CustomerID=user, DSID=ds_product)
                return JsonResponse({"status": "success"})

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)

    return JsonResponse({"status": "error"}, status=400)


@csrf_exempt
def remove_favorite(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            product_id = data.get("product_id")  # DSID
            customer_id = request.session.get('user_id')  # 세션에서 사용자 ID 가져오기

            if customer_id and product_id:
                user = UserProfile.objects.get(CustomerID=customer_id)
                ds_product = DsProduct.objects.get(dsid=product_id)

                # Favorite에서 삭제
                Favorite.objects.filter(CustomerID=user, DSID=ds_product).delete()
                return JsonResponse({"status": "success"})

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)

    return JsonResponse({"status": "error"}, status=400)