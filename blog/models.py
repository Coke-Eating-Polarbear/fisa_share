from django.db import models # type: ignore
from django.contrib.auth.hashers import make_password # type: ignore


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
    stageclass = models.CharField(max_length=1)
    inlevel = models.SmallIntegerField()

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
    

class MyData(models.Model):
    CustomerID = models.CharField(max_length=256, primary_key=True)  # 고객 ID
    pDate = models.DateField()  # 결제 날짜
    bizcode = models.CharField(max_length=256)  # 산업 분류 코드
    store = models.CharField(max_length=256)  # 결제 상호명
    price = models.IntegerField()  # 결제 금액
    Income = models.IntegerField()  # 수입
    Total = models.IntegerField()  # 계좌 잔액
    estate = models.BigIntegerField()  # 부동산 (bigint)
    credit = models.IntegerField()  # 입금 내역
    spend = models.IntegerField()  # 지출

    class Meta:
        db_table = 'mydata'  # 테이블 이름 지정


class Average(models.Model):
    stageclass = models.CharField(max_length=10, primary_key=True)  # StageClass 컬럼
    inlevel = models.IntegerField()  # Inlevel 컬럼
    spend = models.IntegerField()  # 소비
    income = models.IntegerField()  # 수입
    asset = models.IntegerField()  # 자산
    finance = models.IntegerField()  # 금융
    eat = models.IntegerField()  # 식사
    transfer = models.IntegerField()  # 교통
    utility = models.IntegerField()  # 공과금
    phone = models.IntegerField()  # 통신
    home = models.IntegerField()  # 주거
    hobby = models.IntegerField()  # 취미
    fashion = models.IntegerField()  # 패션
    party = models.IntegerField()  # 파티
    allowance = models.IntegerField()  # 용돈
    study = models.IntegerField()  # 학업
    medical = models.IntegerField()  # 의료

    class Meta:
        db_table = 'average'  # 테이블 이름
        managed = False  # Django가 테이블을 생성/수정하지 않도록 설정
        unique_together = (('stageclass', 'inlevel'),)
        constraints = [
            models.UniqueConstraint(fields=['stageclass', 'inlevel'], name='unique_stage_inlevel')
        ]  # 복합 Primary Key 대체로 UniqueConstraint 사용

    def __str__(self):
        return f"{self.stageclass} - {self.inlevel}"

class spend(models.Model):
    CustomerID = models.CharField(max_length=256, primary_key=True)
    SDate = models.DateField()
    Category = models.CharField(max_length=256)
    Frequency = models.IntegerField()
    Amount = models.BigIntegerField()
    store = models.CharField(max_length=256)
    bizCode = models.CharField(max_length=256)

    class Meta:
        db_table = 'spend'

class card(models.Model):
    cardID = models.CharField(max_length=256, primary_key=True)
    cardName = models.CharField(max_length=256)
    benefits = models.CharField(max_length=256)
    image = models.CharField(max_length=256)
    details = models.TextField()
    url = models.CharField(max_length=256)
    cardType = models.CharField(max_length=1)

    class Meta:
        db_table = 'card'