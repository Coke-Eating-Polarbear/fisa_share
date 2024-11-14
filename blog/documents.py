# app_name/documents.py
from elasticsearch_dsl import Document, Text, Date, Keyword

class UserFlowLogDocument(Document):
    user_id = Keyword()          # 사용자 ID (고유한 문자열)
    session_id = Keyword()       # 세션 ID
    action = Text()              # 사용자 행동 (예: 페이지 조회, 버튼 클릭)
    page = Text()                # 페이지 이름 또는 URL
    timestamp = Date()           # 행동 발생 시간

    class Index:
        name = 'user_flow_logs'  # 인덱스 이름 설정
        using = 'default'  # default alias 사용

