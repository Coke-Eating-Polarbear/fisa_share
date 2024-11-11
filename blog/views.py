from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from .forms import LoginForm

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

