from django.db import models
from django.contrib.auth.hashers import make_password


# UserProfile 모델 (회원 정보)
class UserProfile(models.Model):
    CustomerID = models.CharField(max_length=100, unique=True, primary_key=True)  # 아이디
    Pw = models.CharField(max_length=100)  # 비밀번호
    Email = models.EmailField(unique=True)  # 이메일
    username = models.CharField(max_length=100)  # 이름
    Birth = models.CharField(max_length=6)  # 생년월일
    SerialNum = models.CharField(max_length=1)  # 주민번호 뒷자리
    Phone = models.CharField(max_length=11)  # 전화번호
    sex = models.CharField(max_length=1, blank=True)  # 성별 (M, F)

    def save(self, *args, **kwargs):
        # 비밀번호가 이미 해시되지 않은 경우에만 해시화
        if not self.Pw.startswith('pbkdf2_'):
            self.Pw = make_password(self.Pw)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.CustomerID

    class Meta:
        db_table = 'usertable'


# Recommend 모델 (추천 상품 정보)
class Recommend(models.Model):
    CustomerID = models.ForeignKey(UserProfile, on_delete=models.CASCADE, db_column='CustomerID')
    DSID = models.ForeignKey('DsProduct', on_delete=models.CASCADE, db_column='DSID')

    class Meta:
        db_table = 'recommend'
        constraints = [
            models.UniqueConstraint(fields=['CustomerID', 'DSID'], name='unique_recommend')
        ]


# DsProduct 모델 (상품 정보)
class DsProduct(models.Model):
    dsid = models.CharField(max_length=256, primary_key=True)
    bank = models.CharField(max_length=256)
    baser = models.TextField()  # 기준 금리
    maxir = models.TextField()  # 최대 금리
    dstype = models.CharField(max_length=256)  # 상품 유형
    dsname = models.CharField(max_length=256)  # 상품 이름

    class Meta:
        db_table = 'ds_product'


# Wc 모델 (워드 클라우드 이미지 저장)
class Wc(models.Model):
    date = models.DateField()  # 날짜
    image = models.BinaryField()  # BLOB 형태의 이미지

    class Meta:
        db_table = 'wc'


# News 모델 (뉴스 정보)
class News(models.Model):
    norder = models.IntegerField()  # 뉴스 순서
    ndate = models.DateField()  # 뉴스 날짜
    title = models.CharField(max_length=256)  # 뉴스 제목
    content = models.TextField()  # 뉴스 내용
    url = models.CharField(max_length=256)  # 뉴스 URL
    summary = models.TextField()  # 뉴스 요약

    class Meta:
        db_table = 'news'


# Favorite 모델 (찜한 상품 정보)
class Favorite(models.Model):
    CustomerID = models.ForeignKey(
        UserProfile,
        on_delete=models.CASCADE,
        db_column='CustomerID'
    )
    DSID = models.ForeignKey(
        DsProduct,
        on_delete=models.CASCADE,
        db_column='DSID'
    )

    class Meta:
        db_table = 'favorite'
        managed = False  # Django가 테이블을 생성/수정하지 않도록 설정
        unique_together = (('CustomerID', 'DSID'),)  # 복합 기본 키 설정
        constraints = [
            models.UniqueConstraint(fields=['CustomerID', 'DSID'], name='unique_favorite')
        ]

    def __str__(self):
        return f"{self.CustomerID.CustomerID} - {self.DSID.dsid}"