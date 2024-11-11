from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from .forms import LoginForm, UserProfileForm  # UserProfileForm을 가져옵니다
from .models import UserProfile  # UserProfile 모델도 가져옵니다

def main(request):
    return render(request, 'main.html')

def report(request):
    return render(request, 'report.html')

def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data.get('email')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=email, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')  # 로그인 후 이동할 페이지
            else:
                messages.error(request, '없는 아이디거나 틀린 비밀번호입니다.')
    else:
        form = LoginForm()

    return render(request, 'login.html', {'form': form})


def register(request):
    if request.method == "POST":
        form = UserProfileForm(request.POST)
        if form.is_valid():
            # 폼 데이터가 유효하면 데이터베이스에 저장
            user_profile = form.save(commit=False)  # commit=False로 임시 저장
            user_profile.sex = form.cleaned_data['sex']  # 성별 필드 설정
            user_profile.save()  # 데이터베이스에 최종 저장
            messages.success(request, "회원가입이 완료되었습니다.")
            return redirect("login")  # 로그인 페이지로 리디렉션
    else:
        form = UserProfileForm()
    
    return render(request, "register.html", {"form": form})
