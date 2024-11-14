from django.apps import AppConfig
# app_name/apps.py
from django.apps import AppConfig
from elasticsearch_dsl import connections
import os
from dotenv import load_dotenv

load_dotenv()

class BlogConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "blog"


class AppNameConfig(AppConfig):
    name = 'blog'  # 실제 앱 이름으로 변경

    def ready(self):
        # 'default' 연결을 생성하여 Elasticsearch에 연결
        connections.create_connection(alias='default', hosts=[os.getenv('ES')])
