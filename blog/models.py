from django.db import models

# Create your models here.

class UserProfile(models.Model):
    customer_id = models.AutoField(primary_key=True)  # 자동 증가하는 고유 ID
    name = models.CharField(max_length=100)
    user_id = models.CharField(max_length=100, unique=True)
    password = models.CharField(max_length=100)
    birth = models.DateField()  # 날짜 필드
    serial = models.CharField(max_length=7)
    sex = models.CharField(max_length=1, choices=[('M', 'Male'), ('F', 'Female')])  # 성별 필드 추가
    phone = models.CharField(max_length=15, blank=True, null=True)  # 전화번호 필드 추가
    email = models.EmailField(unique=True)

    def __str__(self):
        return self.user_id
    
    class Meta:
        db_table = 'manduck"."usertable' 