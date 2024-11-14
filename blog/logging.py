# 테스트 
# app_name/logging.py
from datetime import datetime
from .documents import UserFlowLogDocument  # documents.py에서 UserFlowLogDocument 가져오기

def log_user_action(user_id, session_id, action, page):
    """
    사용자의 행동을 Elasticsearch에 로그로 저장하는 함수.

    Args:
        user_id (str): 사용자 ID 또는 익명 사용자 구분용 ID.
        session_id (str): 세션 ID.
        action (str): 사용자 행동 (예: 페이지 조회, 버튼 클릭).
        page (str): 페이지 이름 또는 URL.
    """
    log = UserFlowLogDocument(
        user_id=user_id,
        session_id=session_id,
        action=action,
        page=page,
        timestamp=datetime.utcnow()
    )
    log.save()  # Elasticsearch에 로그 저장
