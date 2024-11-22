import json
import random
import string
from django.shortcuts import render, redirect  # 페이지 렌더링 및 리디렉션
from django.contrib.auth.hashers import check_password  # 비밀번호 확인
from blog.models import UserProfile
from blog.forms import UserProfileForm  # UserProfileForm 폼 클래스
from django.http import JsonResponse
from django.contrib.auth.hashers import make_password
from django.core.mail import send_mail, BadHeaderError
from django.utils import timezone # type: ignore
import datetime
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

def map_person(age, marital_status, children=None, children_age=None):
    """
    조건에 따라 사람 유형을 매핑합니다.
    """
    if age >= 20 and age < 30 and marital_status == "미혼":
        return "A"
    elif age >= 30 and age < 50 and marital_status == "미혼":
        return "B"
    elif age >= 20 and age < 50 and marital_status == "기혼" and not children:
        return "C"
    elif age >= 20 and age < 40 and marital_status == "기혼" and children and children_age == "초등생":
        return "D"
    elif age >= 40 and age < 50 and marital_status == "기혼" and children and children_age == "초등생":
        return "E"
    elif age >= 40 and age < 50 and marital_status == "기혼" and children and children_age == "중고등생":
        return "F"
    elif age >= 50 and age < 60 and marital_status == "기혼" and children and children_age in ["중고등생", "대학생"]:
        return "G"
    elif age >= 50 and age < 60 and marital_status == "기혼" and children and children_age == "성인자녀":
        return "H"
    elif age >= 60:
        return "I"
    else:
        return "A"  # 기본값

def signup(request):
    if request.method == 'POST':
        form = UserProfileForm(request.POST)
        if form.is_valid():
            user_profile = form.save(commit=False)

            # 성별 설정
            user_profile.sex = 'M' if user_profile.SerialNum in ['1', '3'] else 'F'

            # 사용자 입력 데이터 가져오기
            marriage_status = request.POST.get('marriage-status', 'N')
            children_status = request.POST.get('children-status', 'N')
            children_age = request.POST.get('children-age', '')

            # 자녀 여부를 Boolean 값으로 변환
            has_children = True if children_status == 'Y' else False

            # 나이 계산 (생년월일 기반)
            birth_year = int(user_profile.Birth[:4])  # 주민번호 앞자리에서 연도 추출
            current_year = datetime.datetime.now().year
            age = current_year - birth_year

            # 매핑
            user_profile.category = map_person(
                age=age,
                marital_status="기혼" if marriage_status == 'Y' else "미혼",
                children=has_children,
                children_age=children_age
            )

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

