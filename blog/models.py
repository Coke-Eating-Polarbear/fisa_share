from django.db import models

class UserProfile(models.Model):
    CustomerID = models.CharField(max_length=100, unique=True)  # 아이디
    Pw = models.CharField(max_length=100)                       # 비밀번호
    Email = models.EmailField(unique=True)                      # 이메일
    username = models.CharField(max_length=100)                 # 이름
    Birth = models.CharField(max_length=6)                      # 생년월일
    SerialNum = models.CharField(max_length=1)                  # 주민번호 뒷자리
    Phone = models.CharField(max_length=11)                     # 전화번호
    sex = models.CharField(max_length=1, blank=True)            # 성별 (M, F)

    def __str__(self):
        return self.CustomerID

    class Meta:
        db_table = 'usertable'
