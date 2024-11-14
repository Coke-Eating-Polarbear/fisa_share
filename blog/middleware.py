# myapp/middleware.py
from django.utils.deprecation import MiddlewareMixin
from django.contrib.auth.middleware import get_user
import datetime
from elasticsearch import Elasticsearch
from .logging import log_user_action
import os
from dotenv import load_dotenv

load_dotenv()

# Elasticsearch 연결 설정
es = Elasticsearch(hosts=[os.getenv('ES')])

class LogOnlyLoggedInMiddleware(MiddlewareMixin):
    def process_request(self, request):
        user = get_user(request)
        
        # 로그인된 사용자만 로그 저장
        if user.is_authenticated:
            self.save_log_to_elasticsearch(user, request)

    def save_log_to_elasticsearch(self, user, request):
        log_data = {
            "user_id": user.id,
            "username": user.username,
            "path": request.path,
            "method": request.method,
            "timestamp": datetime.datetime.now(),
        }
        es.index(index="user_logs", body=log_data)



class UserActionLoggingMiddleware(MiddlewareMixin):
    def process_view(self, request, view_func, view_args, view_kwargs):
        user = get_user(request)
        
        if user.is_authenticated:
            user_id = user.id
        else:
            user_id = "anonymous"  # 로그인하지 않은 사용자 처리
        
        # 세션 ID와 현재 페이지 정보 수집
        session_id = request.session.session_key
        page = request.path
        action = "page_view"

        # 로그 기록 함수 호출
        log_user_action(user_id=user_id, session_id=session_id, action=action, page=page)
