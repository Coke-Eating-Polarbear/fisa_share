import json
import random
import string
from django.shortcuts import render, redirect  # 페이지 렌더링 및 리디렉션
from django.contrib.auth.hashers import check_password  # 비밀번호 확인
from blog.models import UserProfile, MyDataAsset
from blog.forms import UserProfileForm  # UserProfileForm 폼 클래스
from django.http import JsonResponse
from django.contrib.auth.hashers import make_password
from django.core.mail import send_mail, BadHeaderError
from django.utils import timezone # type: ignore
import datetime
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
import os
from tensorflow.keras.models import load_model # type: ignore
import joblib
import pandas as pd # type: ignore
from django.conf import settings


# Create your views here.

def generate_temp_password(length=8):
    """임시 비밀번호 생성 함수"""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def check_user_id(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)  # 요청 데이터 파싱
            user_id = data.get("userId")  # 입력된 아이디 가져오기
            # CustomerID 필드를 기준으로 중복 체크
            if UserProfile.objects.filter(CustomerID=user_id).exists():
                return JsonResponse({"exists": True})  # 이미 존재하는 아이디
            else:
                return JsonResponse({"exists": False})  # 사용 가능한 아이디
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    return JsonResponse({"error": "Invalid request method"}, status=405)

def map_person(age, sex, monthly_income, Financial, debt, total_income):
    """
    조건에 따라 사람 유형을 매핑합니다. -> 근데 체크박스에 있는 것들은 제외한 값들에 대해 모델 인풋
    """
    # if age >= 20 and age < 30 and marital_status == "미혼":
    #     return "A"
    # elif age >= 30 and age < 50 and marital_status == "미혼":
    #     return "B"
    # elif age >= 20 and age < 50 and marital_status == "기혼" and not children:
    #     return "C"
    # elif age >= 20 and age < 40 and marital_status == "기혼" and children and children_age == "초등생":
    #     return "D"
    # elif age >= 40 and age < 50 and marital_status == "기혼" and children and children_age == "초등생":
    #     return "E"
    # elif age >= 40 and age < 50 and marital_status == "기혼" and children and children_age == "중고등생":
    #     return "F"
    # elif age >= 50 and age < 60 and marital_status == "기혼" and children and children_age in ["중고등생", "대학생"]:
    #     return "G"
    # elif age >= 50 and age < 60 and marital_status == "기혼" and children and children_age == "성인자녀":
    #     return "H"
    # elif age >= 60:
    #     return "I"
    # else:
    #     return "A"  # 기본값
    # 딕셔너리 형태로 데이터 매핑
    data = {
        "나이": [age],
        "성별": [sex],
        "월평균소득": [monthly_income],
        "금융자산": [Financial],
        "부채": [debt],
        "총소득": [total_income]
    }
    # 데이터프레임 생성
    X_data = pd.DataFrame(data)
    # 스케일링 (RobustScaler 사용)
    scaler = RobustScaler()
    X_data=scaler.fit_transform(X_data)
    model_path = os.path.join(settings.BASE_DIR, 'models', 'customer_income_no_family_model.keras')
    loaded_model = load_model(model_path)
    # 예측
    predictions = loaded_model.predict(X_data)
    customer_class_pred = predictions[0]  # Customer_Class 예측 (softmax 확률)
    income_group_pred = predictions[1]   # Income_Group 예측 (softmax 확률)

    # 가장 높은 확률의 클래스 선택
    customer_class_result = customer_class_pred.argmax(axis=-1)  # 클래스 번호
    income_group_result = income_group_pred.argmax(axis=-1)      # 클래스 번호
    customer_class_result = int(customer_class_result[0])
    income_group_result = int(income_group_result[0])

    print("클래스 선택 : " ,customer_class_result,income_group_result )
    return customer_class_result, income_group_result



def signup(request):
    print(f"Request Method: {request.method}")
    print(f"Request Data: {request.POST}")
    if request.method == 'POST':
        print("POST request detected")
        ## 추가한 부분
         # POST 데이터 수정: email과 email-domain을 합쳐 Email 필드로 생성
        email_id = request.POST.get('email', '')
        email_domain = request.POST.get('email-domain', '')
        
        # 사용자 입력 데이터로 Email 필드 생성
        if email_domain == 'custom':
            email_domain = request.POST.get('custom-email-domain', '')
        
        # Email 데이터 조합
        request.POST = request.POST.copy()  # POST 데이터는 불변이므로 복사 필요
        request.POST['Email'] = f"{email_id}@{email_domain}"
        form = UserProfileForm(request.POST)
        print(f"form print : {form}" )
        if form.is_valid():
            user_profile = form.save(commit=False)
            print("user_profile", user_profile)
            print(form)
            print(user_profile)

            # 성별 설정
            user_profile.sex = 'M' if user_profile.SerialNum in ['1', '3'] else 'F'
            sex = int(1) if user_profile.sex == 'M' else int(2)

            
            # 나이 계산 (생년월일 기반)
            birth_year = (1900+int(user_profile.Birth[:2])) if user_profile.SerialNum in ['1', '2'] else (2000 +int(user_profile.Birth[:2]))  # 주민번호 앞자리에서 연도 추출
            current_year = datetime.datetime.now().year
            age = current_year - birth_year
            # mydata_asset에서 들고옴
            # 데이터를 하나 가져오는 예제
            data = MyDataAsset.objects.first()
            if data:
                monthly_income = data.monthly_income
                financial = data.financial
                debt = data.debt
                total_income = data.total_income
                
            else:
                print("No data found in mydata_asset table.")

            # 매핑
            customer_class_pred, income_group_pred = map_person(
                age=age,
                sex=sex,
                monthly_income= monthly_income,
                Financial= financial,
                debt= debt, 
                total_income = total_income
            )
            user_profile.Stageclass = customer_class_pred
            user_profile.Inlevel = income_group_pred
            print(user_profile)

            # 데이터베이스 저장
            user_profile.save()
            return redirect('accounts:login')  # 로그인 페이지로 리다이렉트
        else:
            print(form.errors)  # 폼 에러 출력
            return render(request, 'accounts/signup.html', {'form': form})
    else:
        form = UserProfileForm()
    return render(request, 'accounts/signup.html', {'form': form})

def agree(request):
    return render(request, 'accounts/join_agree.html')

def login_view(request):  # 이름을 login_view로 변경
    if request.method == 'POST':
        customer_id = request.POST['CustomerID']
        password = request.POST['Pw']

        try:
            # UserProfile에서 CustomerID를 이용하여 사용자 객체 가져오기
            user = UserProfile.objects.get(CustomerID=customer_id)
            
            # 비밀번호 확인
            if check_password(password, user.Pw):
                # 세션에 사용자 정보 저장하기
                request.session['user_id'] = user.CustomerID
                return redirect('loginmain')  # main.html로 리디렉션
            else:
                # 비밀번호가 틀린 경우
                return render(request, 'accounts/login.html', {'error': 'Invalid ID or password.'})
        
        except UserProfile.DoesNotExist:
            # 사용자 ID가 없는 경우
            return render(request, 'accounts/login.html', {'error': 'Invalid ID or password.'})
    
    return render(request, 'accounts/login.html')

def findid(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            name = data.get('name')
            email = data.get('email')
            Phone = data.get('Phone')
            # 사용자 데이터 확인
            user = UserProfile.objects.filter(
                username__iexact=name,
                Email__iexact=email,
                Phone__iexact=Phone
            ).first()

            if user:
                # 이메일로 ID 전송
                try:
                    send_mail(
                        "Your ID",
                        f"안녕하세요, {name}님! 회원님의 ID는 {user.CustomerID}입니다.",
                        "noreply@gmail.com",
                        [email],
                        fail_silently=False,
                    )
                    print("Email sent successfully")
                except Exception as e:
                    print(f"Failed to send email: {e}")
                return JsonResponse({"success": True})
            else:
                print("No matching user found")
                return JsonResponse({"success": False, "error": "일치하는 사용자가 없습니다."})
        except Exception as e:
            print("Error:", e)
            return JsonResponse({"success": False, "error": str(e)})
    return render(request, 'accounts/findid.html')

def findpw(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            name = data.get('name')
            email = data.get('email')
            Phone = data.get('Phone')
            CustomerID = data.get('CustomerID')

            # 사용자 데이터 확인
            user = UserProfile.objects.filter(
                username__iexact=name,
                Email__iexact=email,
                Phone__iexact=Phone,
                CustomerID__iexact=CustomerID
            ).first()

            if user:
                # 임시 비밀번호 생성
                temp_password = generate_temp_password()

                # 데이터베이스에 암호화된 비밀번호 저장
                user.Pw = make_password(temp_password)
                user.save()

                # 이메일로 임시 비밀번호 전송
                try:
                    send_mail(
                        "Your Temporary Password",
                        f"안녕하세요, {name}님! 임시 비밀번호는 {temp_password}입니다. 로그인 후 반드시 비밀번호를 변경해주세요.",
                        "noreply@gmail.com",
                        [email],
                        fail_silently=False,
                    )
                    print("Temporary password sent successfully")
                except Exception as e:
                    print(f"Failed to send email: {e}")
                    return JsonResponse({"success": False, "error": "임시 비밀번호 이메일 전송 실패"})

                return JsonResponse({"success": True, "message": "임시 비밀번호가 이메일로 전송되었습니다."})
            else:
                print("No matching user found")
                return JsonResponse({"success": False, "error": "일치하는 사용자가 없습니다."})

        except Exception as e:
            print("Error:", e)
            return JsonResponse({"success": False, "error": str(e)})
    
    return render(request, 'accounts/findpw.html')




def terms_content2(request):
    return render(request, 'accounts/mydata_form2.html')

def terms_content3(request):
    return render(request, 'accounts/mydata_form3.html')

def terms_content4(request):
    return render(request, 'accounts/logdata_form.html')

def terms_content5(request):
    return render(request, 'accounts/marketing.html')

