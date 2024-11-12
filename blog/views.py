from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from .forms import UserProfileForm  # UserProfileForm을 가져옵니다
from .models import UserProfile  # UserProfile 모델도 가져옵니다

def main(request):
    return render(request, 'main.html')

def report(request):
    return render(request, 'report.html')

def agree(request):
    return render(request, 'join_agree.html')

def login(request):
    return render(request, 'login.html')

def signup(request):
    if request.method == 'POST':
        form = UserProfileForm(request.POST)
        if form.is_valid():
            user_profile = form.save(commit=False)
            user_profile.sex = 'M' if user_profile.SerialNum in ['1', '3'] else 'F'
            print(form.cleaned_data)  # 로그 출력
            user_profile.save()  # 데이터베이스에 저장
            return redirect('login')
        else:
            print(form.errors)  # 폼 에러 출력
    else:
        form = UserProfileForm()
    return render(request, 'signup.html', {'form': form})
