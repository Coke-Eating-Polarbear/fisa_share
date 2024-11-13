from django.db import models
from django.contrib.auth.hashers import make_password

class UserProfile(models.Model):
    CustomerID = models.CharField(max_length=100, unique=True,primary_key=True)  # 아이디
    Pw = models.CharField(max_length=100)                       # 비밀번호
    Email = models.EmailField(unique=True)                      # 이메일
    username = models.CharField(max_length=100)                 # 이름
    Birth = models.CharField(max_length=6)                      # 생년월일
    SerialNum = models.CharField(max_length=1)                  # 주민번호 뒷자리
    Phone = models.CharField(max_length=11)                     # 전화번호
    sex = models.CharField(max_length=1, blank=True)            # 성별 (M, F)

    def save(self, *args, **kwargs):
        # 비밀번호가 이미 해시되지 않은 경우에만 해시화
        if not self.Pw.startswith('pbkdf2_'):
            self.Pw = make_password(self.Pw)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.CustomerID

    class Meta:
        db_table = 'usertable'


class Recommend(models.Model):
    customerid = models.CharField(max_length=256, primary_key=True)
    dsid = models.CharField(max_length=256)

    class Meta:
        db_table = 'recommend'

class DsProduct(models.Model):
    dsid = models.CharField(max_length=256, primary_key=True)
    bank = models.CharField(max_length=256)
    baser = models.TextField()
    maxir = models.TextField()
    dstype = models.CharField(max_length=256)
    dsname = models.CharField(max_length=256)

    class Meta:
        db_table = 'ds_product'